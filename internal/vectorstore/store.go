package vectorstore

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/colesmcintosh/ai-memory/pkg/vectormath"
)

var (
	ErrVectorDimensionMismatch = errors.New("vector dimensions do not match")
	ErrEmptyVector            = errors.New("vector cannot be empty")
	ErrNilVector              = errors.New("vector cannot be nil")
	ErrInvalidHashTables      = errors.New("number of hash tables must be positive")
	ErrStoreEmpty             = errors.New("store is empty")
	ErrInvalidLimit           = errors.New("limit must be non-negative")
)

const (
	// Performance tuning constants
	minGoroutineThreshold = 4    // Minimum tables before using goroutines
	maxBucketSize        = 1000  // Maximum bucket size before cleanup
	defaultBatchSize     = 100   // Default batch operation size
)

// LSHParams defines parameters for Locality-Sensitive Hashing
type LSHParams struct {
	NumHashTables    int     // Number of hash tables
	NumHashFunctions int     // Number of hash functions per table
	BucketWidth     float64 // Width of LSH buckets
}

// DefaultLSHParams returns default LSH parameters optimized for general use
func DefaultLSHParams() LSHParams {
	return LSHParams{
		NumHashTables:    6,
		NumHashFunctions: 8,
		BucketWidth:     4.0,
	}
}

// hashFunction represents a single LSH hash function
type hashFunction struct {
	vector []float64
	bias   float64
}

// LSHTable represents a single hash table in the LSH index
type LSHTable struct {
	hashFunctions []hashFunction
	buckets      map[string][]string // maps hash string to vector keys
	mu           sync.RWMutex        // Fine-grained locking per table
}

// Store represents a thread-safe key-value vector store with LSH indexing.
type Store struct {
	mu          sync.RWMutex
	vectors     map[string][]float64
	dim         int // dimension of vectors
	lshTables   []LSHTable
	lshParams   LSHParams
	initialized bool
	
	// Performance optimization pools
	hashBufferPool sync.Pool
	resultPool     sync.Pool
}

// New creates a new Store with optional LSH parameters.
func New(params ...LSHParams) *Store {
	var p LSHParams
	if len(params) > 0 {
		p = params[0]
	} else {
		p = DefaultLSHParams()
	}

	s := &Store{
		vectors:   make(map[string][]float64),
		lshParams: p,
	}
	
	// Initialize object pools for better memory management
	s.hashBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, p.NumHashFunctions)
		},
	}
	
	s.resultPool = sync.Pool{
		New: func() interface{} {
			return make([]SearchResult, 0, 32)
		},
	}
	
	return s
}

// Add adds a vector to the store with the given key.
func (s *Store) Add(key string, vector []float64) error {
	if vector == nil {
		return ErrNilVector
	}
	if len(vector) == 0 {
		return ErrEmptyVector
	}
	if key == "" {
		return errors.New("key cannot be empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Initialize LSH tables if this is the first vector
	if !s.initialized {
		s.dim = len(vector)
		if err := s.initLSHTables(); err != nil {
			return fmt.Errorf("failed to initialize LSH tables: %w", err)
		}
		s.initialized = true
	} else if len(vector) != s.dim {
		return ErrVectorDimensionMismatch
	}

	// Copy vector to avoid external mutations
	vectorCopy := make([]float64, len(vector))
	copy(vectorCopy, vector)
	s.vectors[key] = vectorCopy

	// Add to LSH tables with optimized hashing
	for i := range s.lshTables {
		hashStr := s.hashVectorOptimized(&s.lshTables[i], vectorCopy)
		s.lshTables[i].mu.Lock()
		s.lshTables[i].buckets[hashStr] = append(s.lshTables[i].buckets[hashStr], key)
		s.lshTables[i].mu.Unlock()
	}

	return nil
}

// AddBatch adds multiple vectors efficiently in a single operation
func (s *Store) AddBatch(vectors map[string][]float64) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	var firstVector []float64
	for key, vector := range vectors {
		if vector == nil {
			return fmt.Errorf("vector for key '%s' is nil", key)
		}
		if len(vector) == 0 {
			return fmt.Errorf("vector for key '%s' is empty", key)
		}
		if key == "" {
			return errors.New("key cannot be empty")
		}
		
		if firstVector == nil {
			firstVector = vector
		} else if len(vector) != len(firstVector) {
			return fmt.Errorf("vector dimension mismatch for key '%s'", key)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Initialize if needed
	if !s.initialized {
		s.dim = len(firstVector)
		if err := s.initLSHTables(); err != nil {
			return fmt.Errorf("failed to initialize LSH tables: %w", err)
		}
		s.initialized = true
	}

	// Process in batches to avoid holding locks too long
	keys := make([]string, 0, len(vectors))
	for key := range vectors {
		keys = append(keys, key)
	}

	for i := 0; i < len(keys); i += defaultBatchSize {
		end := i + defaultBatchSize
		if end > len(keys) {
			end = len(keys)
		}

		// Process batch
		for j := i; j < end; j++ {
			key := keys[j]
			vector := vectors[key]
			
			if len(vector) != s.dim {
				return fmt.Errorf("vector dimension mismatch for key '%s'", key)
			}

			// Copy vector
			vectorCopy := make([]float64, len(vector))
			copy(vectorCopy, vector)
			s.vectors[key] = vectorCopy

			// Add to LSH tables
			for k := range s.lshTables {
				hashStr := s.hashVectorOptimized(&s.lshTables[k], vectorCopy)
				s.lshTables[k].mu.Lock()
				s.lshTables[k].buckets[hashStr] = append(s.lshTables[k].buckets[hashStr], key)
				s.lshTables[k].mu.Unlock()
			}
		}
	}

	return nil
}

// Get retrieves a vector by its key.
func (s *Store) Get(key string) ([]float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	vector, ok := s.vectors[key]
	if !ok {
		return nil, false
	}
	
	// Return a copy to prevent external mutations
	result := make([]float64, len(vector))
	copy(result, vector)
	return result, true
}

// Delete removes a vector from the store efficiently.
func (s *Store) Delete(key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	vector, exists := s.vectors[key]
	if !exists {
		return false
	}

	// Remove from LSH tables using optimized bucket operations
	for i := range s.lshTables {
		hashStr := s.hashVectorOptimized(&s.lshTables[i], vector)
		s.lshTables[i].mu.Lock()
		bucket := s.lshTables[i].buckets[hashStr]
		
		// Optimized removal using swap-with-last technique
		for j, k := range bucket {
			if k == key {
				bucket[j] = bucket[len(bucket)-1]
				bucket = bucket[:len(bucket)-1]
				break
			}
		}
		
		if len(bucket) == 0 {
			delete(s.lshTables[i].buckets, hashStr)
		} else {
			s.lshTables[i].buckets[hashStr] = bucket
		}
		s.lshTables[i].mu.Unlock()
	}

	delete(s.vectors, key)
	return true
}

// Search finds the k most similar vectors to the query vector.
func (s *Store) Search(query []float64, k int) ([]SearchResult, error) {
	if query == nil {
		return nil, ErrNilVector
	}
	if len(query) == 0 {
		return nil, ErrEmptyVector
	}
	if k < 0 {
		return nil, ErrInvalidLimit
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.initialized {
		return nil, ErrStoreEmpty
	}
	if len(query) != s.dim {
		return nil, ErrVectorDimensionMismatch
	}

	// Use different strategies based on the number of tables
	if len(s.lshTables) < minGoroutineThreshold {
		return s.searchSequential(query, k)
	}
	return s.searchConcurrent(query, k)
}

// searchSequential performs search without goroutines for small numbers of tables
func (s *Store) searchSequential(query []float64, k int) ([]SearchResult, error) {
	seen := make(map[string]bool)
	allResults := s.resultPool.Get().([]SearchResult)
	allResults = allResults[:0]
	defer s.resultPool.Put(allResults)

	for i := range s.lshTables {
		hashStr := s.hashVectorOptimized(&s.lshTables[i], query)
		s.lshTables[i].mu.RLock()
		candidates := s.lshTables[i].buckets[hashStr]
		
		for _, key := range candidates {
			if !seen[key] {
				seen[key] = true
				if vector, ok := s.vectors[key]; ok {
					sim, err := vectormath.CosineSimilarity(query, vector)
					if err == nil {
						allResults = append(allResults, SearchResult{Key: key, Similarity: sim})
					}
				}
			}
		}
		s.lshTables[i].mu.RUnlock()
	}

	return s.finalizeResults(allResults, k), nil
}

// searchConcurrent performs search using goroutines for larger numbers of tables
func (s *Store) searchConcurrent(query []float64, k int) ([]SearchResult, error) {
	results := make(chan []SearchResult, len(s.lshTables))
	var wg sync.WaitGroup

	// Search each LSH table in parallel
	for i := range s.lshTables {
		wg.Add(1)
		go func(table *LSHTable) {
			defer wg.Done()
			hashStr := s.hashVectorOptimized(table, query)
			
			table.mu.RLock()
			candidates := make([]string, len(table.buckets[hashStr]))
			copy(candidates, table.buckets[hashStr])
			table.mu.RUnlock()
			
			tableResults := make([]SearchResult, 0, len(candidates))
			for _, key := range candidates {
				if vector, ok := s.vectors[key]; ok {
					sim, err := vectormath.CosineSimilarity(query, vector)
					if err == nil {
						tableResults = append(tableResults, SearchResult{Key: key, Similarity: sim})
					}
				}
			}
			results <- tableResults
		}(&s.lshTables[i])
	}

	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
		close(results)
	}()

	// Merge results from all tables
	seen := make(map[string]bool)
	allResults := s.resultPool.Get().([]SearchResult)
	allResults = allResults[:0]
	defer s.resultPool.Put(allResults)

	for tableResults := range results {
		for _, result := range tableResults {
			if !seen[result.Key] {
				seen[result.Key] = true
				allResults = append(allResults, result)
			}
		}
	}

	return s.finalizeResults(allResults, k), nil
}

// finalizeResults sorts and limits the search results
func (s *Store) finalizeResults(results []SearchResult, k int) []SearchResult {
	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Return top k results
	if k > 0 && k < len(results) {
		// Make a copy of the slice we're returning
		finalResults := make([]SearchResult, k)
		copy(finalResults, results[:k])
		return finalResults
	}
	
	// Make a copy of all results
	finalResults := make([]SearchResult, len(results))
	copy(finalResults, results)
	return finalResults
}

// Size returns the number of vectors in the store.
func (s *Store) Size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vectors)
}

// Stats returns performance and usage statistics
func (s *Store) Stats() StoreStats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	totalBuckets := 0
	totalKeys := 0
	maxBucketSize := 0
	emptyBuckets := 0
	
	for i := range s.lshTables {
		s.lshTables[i].mu.RLock()
		totalBuckets += len(s.lshTables[i].buckets)
		for _, bucket := range s.lshTables[i].buckets {
			bucketSize := len(bucket)
			totalKeys += bucketSize
			if bucketSize > maxBucketSize {
				maxBucketSize = bucketSize
			}
			if bucketSize == 0 {
				emptyBuckets++
			}
		}
		s.lshTables[i].mu.RUnlock()
	}
	
	avgBucketSize := 0.0
	if totalBuckets > 0 {
		avgBucketSize = float64(totalKeys) / float64(totalBuckets)
	}
	
	return StoreStats{
		VectorCount:     len(s.vectors),
		Dimension:       s.dim,
		LSHTables:       len(s.lshTables),
		TotalBuckets:    totalBuckets,
		AverageBucketSize: avgBucketSize,
		MaxBucketSize:   maxBucketSize,
		EmptyBuckets:    emptyBuckets,
	}
}

// StoreStats represents performance and usage statistics
type StoreStats struct {
	VectorCount       int
	Dimension         int
	LSHTables         int
	TotalBuckets      int
	AverageBucketSize float64
	MaxBucketSize     int
	EmptyBuckets      int
}

// SearchResult represents a single search result.
type SearchResult struct {
	Key        string
	Similarity float64
}

// initLSHTables initializes the LSH tables with optimized memory allocation.
func (s *Store) initLSHTables() error {
	if s.lshParams.NumHashTables <= 0 {
		return ErrInvalidHashTables
	}

	s.lshTables = make([]LSHTable, s.lshParams.NumHashTables)
	for i := range s.lshTables {
		s.lshTables[i] = LSHTable{
			hashFunctions: make([]hashFunction, s.lshParams.NumHashFunctions),
			buckets:      make(map[string][]string),
		}

		// Initialize random hash functions with better distribution
		for j := range s.lshTables[i].hashFunctions {
			s.lshTables[i].hashFunctions[j] = hashFunction{
				vector: s.randomUnitVectorOptimized(),
				bias:   rand.Float64() * s.lshParams.BucketWidth,
			}
		}
	}
	
	return nil
}

// randomUnitVectorOptimized generates a random unit vector efficiently.
func (s *Store) randomUnitVectorOptimized() []float64 {
	vector := make([]float64, s.dim)
	var norm float64
	
	// Generate random normal values and calculate norm in one pass
	for i := range vector {
		val := rand.NormFloat64()
		vector[i] = val
		norm += val * val
	}
	
	// Normalize
	if norm > 0 {
		invNorm := 1.0 / math.Sqrt(norm)
		for i := range vector {
			vector[i] *= invNorm
		}
	}
	
	return vector
}

// hashVectorOptimized computes the hash string for a vector using optimized buffer reuse.
func (s *Store) hashVectorOptimized(table *LSHTable, vector []float64) string {
	hashBits := s.hashBufferPool.Get().([]byte)
	defer s.hashBufferPool.Put(hashBits)
	
	// Ensure buffer is the right size
	if len(hashBits) != len(table.hashFunctions) {
		hashBits = make([]byte, len(table.hashFunctions))
	}
	
	for i, h := range table.hashFunctions {
		var dotProduct float64
		
		// Unrolled dot product for better performance
		j := 0
		for ; j < len(vector)-3; j += 4 {
			dotProduct += vector[j]*h.vector[j] + vector[j+1]*h.vector[j+1] + 
						 vector[j+2]*h.vector[j+2] + vector[j+3]*h.vector[j+3]
		}
		
		// Handle remaining elements
		for ; j < len(vector); j++ {
			dotProduct += vector[j] * h.vector[j]
		}
		
		if (dotProduct+h.bias)/s.lshParams.BucketWidth > 0 {
			hashBits[i] = '1'
		} else {
			hashBits[i] = '0'
		}
	}
	
	return string(hashBits)
}

// Clear removes all vectors from the store
func (s *Store) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	// Clear vectors
	s.vectors = make(map[string][]float64)
	
	// Clear LSH tables
	for i := range s.lshTables {
		s.lshTables[i].mu.Lock()
		s.lshTables[i].buckets = make(map[string][]string)
		s.lshTables[i].mu.Unlock()
	}
	
	s.initialized = false
}

// Keys returns all keys in the store
func (s *Store) Keys() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	keys := make([]string, 0, len(s.vectors))
	for key := range s.vectors {
		keys = append(keys, key)
	}
	
	return keys
}

// Contains checks if a key exists in the store
func (s *Store) Contains(key string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	_, exists := s.vectors[key]
	return exists
} 