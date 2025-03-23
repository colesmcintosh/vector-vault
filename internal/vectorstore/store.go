package vectorstore

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"sync"
)

var (
	ErrVectorDimensionMismatch = errors.New("vector dimensions do not match")
	ErrEmptyVector            = errors.New("vector cannot be empty")
	ErrNilVector              = errors.New("vector cannot be nil")
	ErrInvalidHashTables      = errors.New("number of hash tables must be positive")
)

// LSHParams defines parameters for Locality-Sensitive Hashing
type LSHParams struct {
	NumHashTables    int     // Number of hash tables
	NumHashFunctions int     // Number of hash functions per table
	BucketWidth     float64 // Width of LSH buckets
}

// DefaultLSHParams returns default LSH parameters
func DefaultLSHParams() LSHParams {
	return LSHParams{
		NumHashTables:    4,
		NumHashFunctions: 6,
		BucketWidth:     4.0,
	}
}

// hashFunction represents a single LSH hash function
type hashFunction struct {
	vector    []float64
	bias     float64
}

// LSHTable represents a single hash table in the LSH index
type LSHTable struct {
	hashFunctions []hashFunction
	buckets      map[string][]string // maps hash string to vector keys
}

// Store represents a thread-safe key-value vector store with LSH indexing.
type Store struct {
	mu          sync.RWMutex
	vectors     map[string][]float64
	dim         int // dimension of vectors
	lshTables   []LSHTable
	lshParams   LSHParams
	initialized bool
}

// New creates a new Store with optional LSH parameters.
func New(params ...LSHParams) *Store {
	var p LSHParams
	if len(params) > 0 {
		p = params[0]
	} else {
		p = DefaultLSHParams()
	}

	return &Store{
		vectors:   make(map[string][]float64),
		lshParams: p,
	}
}

// Add adds a vector to the store with the given key.
func (s *Store) Add(key string, vector []float64) error {
	if vector == nil {
		return ErrNilVector
	}
	if len(vector) == 0 {
		return ErrEmptyVector
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Initialize LSH tables if this is the first vector
	if !s.initialized {
		s.dim = len(vector)
		s.initLSHTables()
		s.initialized = true
	} else if len(vector) != s.dim {
		return ErrVectorDimensionMismatch
	}

	// Store the vector
	s.vectors[key] = vector

	// Add to LSH tables
	for i := range s.lshTables {
		hashStr := s.hashVector(&s.lshTables[i], vector)
		s.lshTables[i].buckets[hashStr] = append(s.lshTables[i].buckets[hashStr], key)
	}

	return nil
}

// Get retrieves a vector by its key.
func (s *Store) Get(key string) ([]float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	vector, ok := s.vectors[key]
	return vector, ok
}

// Delete removes a vector from the store.
func (s *Store) Delete(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Remove from LSH tables
	vector, exists := s.vectors[key]
	if !exists {
		return
	}

	for i := range s.lshTables {
		hashStr := s.hashVector(&s.lshTables[i], vector)
		bucket := s.lshTables[i].buckets[hashStr]
		for j, k := range bucket {
			if k == key {
				// Remove the key from the bucket
				bucket = append(bucket[:j], bucket[j+1:]...)
				break
			}
		}
		if len(bucket) == 0 {
			delete(s.lshTables[i].buckets, hashStr)
		} else {
			s.lshTables[i].buckets[hashStr] = bucket
		}
	}

	// Remove from vectors map
	delete(s.vectors, key)
}

// Search finds the k most similar vectors to the query vector.
func (s *Store) Search(query []float64, k int) ([]SearchResult, error) {
	if query == nil {
		return nil, ErrNilVector
	}
	if len(query) == 0 {
		return nil, ErrEmptyVector
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.initialized {
		return nil, errors.New("store is empty")
	}
	if len(query) != s.dim {
		return nil, ErrVectorDimensionMismatch
	}

	results := make(chan []SearchResult, s.lshParams.NumHashTables)
	var wg sync.WaitGroup

	// Search each LSH table in parallel
	for i := range s.lshTables {
		wg.Add(1)
		go func(table *LSHTable) {
			defer wg.Done()
			hashStr := s.hashVector(table, query)
			candidates := table.buckets[hashStr]
			tableResults := make([]SearchResult, 0, len(candidates))

			for _, key := range candidates {
				if vector, ok := s.vectors[key]; ok {
					sim, _ := CosineSimilarity(query, vector)
					tableResults = append(tableResults, SearchResult{Key: key, Similarity: sim})
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
	allResults := make([]SearchResult, 0)
	for tableResults := range results {
		for _, result := range tableResults {
			if !seen[result.Key] {
				seen[result.Key] = true
				allResults = append(allResults, result)
			}
		}
	}

	// Sort by similarity
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Similarity > allResults[j].Similarity
	})

	// Return top k results
	if k > 0 && k < len(allResults) {
		return allResults[:k], nil
	}
	return allResults, nil
}

// Size returns the number of vectors in the store.
func (s *Store) Size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vectors)
}

// SearchResult represents a single search result.
type SearchResult struct {
	Key        string
	Similarity float64
}

// CosineSimilarity calculates the cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, ErrVectorDimensionMismatch
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, nil
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// initLSHTables initializes the LSH tables.
func (s *Store) initLSHTables() {
	if s.lshParams.NumHashTables <= 0 {
		s.lshParams = DefaultLSHParams()
	}

	s.lshTables = make([]LSHTable, s.lshParams.NumHashTables)
	for i := range s.lshTables {
		s.lshTables[i] = LSHTable{
			hashFunctions: make([]hashFunction, s.lshParams.NumHashFunctions),
			buckets:      make(map[string][]string),
		}

		// Initialize random hash functions
		for j := range s.lshTables[i].hashFunctions {
			s.lshTables[i].hashFunctions[j] = hashFunction{
				vector: s.randomUnitVector(),
				bias:   rand.Float64() * s.lshParams.BucketWidth,
			}
		}
	}
}

// randomUnitVector generates a random unit vector.
func (s *Store) randomUnitVector() []float64 {
	vector := make([]float64, s.dim)
	var norm float64
	for i := range vector {
		vector[i] = rand.NormFloat64()
		norm += vector[i] * vector[i]
	}
	norm = math.Sqrt(norm)
	for i := range vector {
		vector[i] /= norm
	}
	return vector
}

// hashVector computes the hash string for a vector using the given LSH table.
func (s *Store) hashVector(table *LSHTable, vector []float64) string {
	hashBits := make([]byte, len(table.hashFunctions))
	for i, h := range table.hashFunctions {
		var dotProduct float64
		for j := range vector {
			dotProduct += vector[j] * h.vector[j]
		}
		if (dotProduct + h.bias) / s.lshParams.BucketWidth > 0 {
			hashBits[i] = '1'
		} else {
			hashBits[i] = '0'
		}
	}
	return string(hashBits)
} 