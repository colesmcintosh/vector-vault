package vectorstore

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/colesmcintosh/ai-memory/pkg/vectormath"
)

func TestNewVectorStore(t *testing.T) {
	tests := []struct {
		name   string
		params LSHParams
	}{
		{
			name:   "default parameters",
			params: DefaultLSHParams(),
		},
		{
			name: "custom parameters",
			params: LSHParams{
				NumHashTables:    8,
				NumHashFunctions: 10,
				BucketWidth:     2.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vs := New(tt.params)
			if vs == nil {
				t.Error("New returned nil")
			}
			if vs.vectors == nil {
				t.Error("vectors map not initialized")
			}
			if vs.lshParams != tt.params {
				t.Errorf("got params = %v, want %v", vs.lshParams, tt.params)
			}
			
			// Test that object pools are initialized
			if vs.hashBufferPool.New == nil {
				t.Error("hashBufferPool not initialized")
			}
			if vs.resultPool.New == nil {
				t.Error("resultPool not initialized")
			}
		})
	}
}

func TestVectorStoreAdd(t *testing.T) {
	vs := New(DefaultLSHParams())

	tests := []struct {
		name    string
		key     string
		vector  []float64
		wantErr bool
	}{
		{
			name:    "valid vector",
			key:     "test1",
			vector:  []float64{1, 2, 3},
			wantErr: false,
		},
		{
			name:    "nil vector",
			key:     "test2",
			vector:  nil,
			wantErr: true,
		},
		{
			name:    "empty vector",
			key:     "test3",
			vector:  []float64{},
			wantErr: true,
		},
		{
			name:    "empty key",
			key:     "",
			vector:  []float64{1, 2, 3},
			wantErr: true,
		},
		{
			name:    "wrong dimension",
			key:     "test4",
			vector:  []float64{1, 2},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := vs.Add(tt.key, tt.vector)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr {
				got, ok := vs.Get(tt.key)
				if !ok {
					t.Errorf("vector not found after adding")
				}
				if !vectorEqual(got, tt.vector) {
					t.Errorf("got vector = %v, want %v", got, tt.vector)
				}
				
				// Test that vector was copied (mutation safety)
				if tt.vector != nil && len(tt.vector) > 0 {
					original := tt.vector[0]
					tt.vector[0] = 999
					retrieved, _ := vs.Get(tt.key)
					if retrieved[0] != original {
						t.Error("vector was not properly copied")
					}
				}
			}
		})
	}
}

func TestVectorStoreAddBatch(t *testing.T) {
	vs := New(DefaultLSHParams())

	t.Run("valid batch", func(t *testing.T) {
		vectors := map[string][]float64{
			"vec1": {1, 2, 3},
			"vec2": {4, 5, 6},
			"vec3": {7, 8, 9},
		}

		err := vs.AddBatch(vectors)
		if err != nil {
			t.Fatalf("AddBatch() error = %v", err)
		}

		if vs.Size() != 3 {
			t.Errorf("Size() = %v, want 3", vs.Size())
		}

		for key, expectedVector := range vectors {
			gotVector, ok := vs.Get(key)
			if !ok {
				t.Errorf("vector %s not found", key)
			}
			if !vectorEqual(gotVector, expectedVector) {
				t.Errorf("vector %s = %v, want %v", key, gotVector, expectedVector)
			}
		}
	})

	t.Run("empty batch", func(t *testing.T) {
		err := vs.AddBatch(map[string][]float64{})
		if err != nil {
			t.Errorf("AddBatch() with empty map should not error, got %v", err)
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		vectors := map[string][]float64{
			"vec1": {1, 2, 3},
			"vec2": {4, 5},
		}

		err := vs.AddBatch(vectors)
		if err == nil {
			t.Error("AddBatch() should error on dimension mismatch")
		}
	})

	t.Run("nil vector in batch", func(t *testing.T) {
		vectors := map[string][]float64{
			"vec1": {1, 2, 3},
			"vec2": nil,
		}

		err := vs.AddBatch(vectors)
		if err == nil {
			t.Error("AddBatch() should error on nil vector")
		}
	})

	t.Run("large batch", func(t *testing.T) {
		vs := New(DefaultLSHParams())
		vectors := make(map[string][]float64)
		
		// Create a batch larger than defaultBatchSize
		for i := 0; i < defaultBatchSize*3; i++ {
			key := fmt.Sprintf("vec%d", i)
			vectors[key] = []float64{float64(i), float64(i + 1), float64(i + 2)}
		}

		err := vs.AddBatch(vectors)
		if err != nil {
			t.Fatalf("AddBatch() error = %v", err)
		}

		if vs.Size() != len(vectors) {
			t.Errorf("Size() = %v, want %v", vs.Size(), len(vectors))
		}
	})
}

func TestVectorStoreDelete(t *testing.T) {
	vs := New(DefaultLSHParams())
	vector := []float64{1, 2, 3}
	key := "test"

	// Add a vector
	err := vs.Add(key, vector)
	if err != nil {
		t.Fatalf("failed to add vector: %v", err)
	}

	// Delete the vector
	deleted := vs.Delete(key)
	if !deleted {
		t.Error("Delete() should return true for existing key")
	}

	// Verify deletion
	_, ok := vs.Get(key)
	if ok {
		t.Error("vector still exists after deletion")
	}

	// Delete non-existent key
	deleted = vs.Delete("nonexistent")
	if deleted {
		t.Error("Delete() should return false for non-existent key")
	}
}

func TestVectorStoreSearch(t *testing.T) {
	vs := New(DefaultLSHParams())

	// Add some test vectors
	vectors := map[string][]float64{
		"vec1": {1, 0, 0},
		"vec2": {0, 1, 0},
		"vec3": {0, 0, 1},
		"vec4": {1, 1, 1},
	}

	for k, v := range vectors {
		if err := vs.Add(k, v); err != nil {
			t.Fatalf("failed to add vector %s: %v", k, err)
		}
	}

	tests := []struct {
		name        string
		query       []float64
		limit       int
		wantFirst   string
		minResults  int
		wantErr     bool
	}{
		{
			name:        "exact match x-axis",
			query:       []float64{1, 0, 0},
			limit:       1,
			wantFirst:   "vec1",
			minResults:  1,
			wantErr:     false,
		},
		{
			name:        "similar to diagonal",
			query:       []float64{1, 1, 0.9},
			limit:       2,
			wantFirst:   "vec4",
			minResults:  2,
			wantErr:     false,
		},
		{
			name:        "all results",
			query:       []float64{1, 1, 1},
			limit:       0,
			wantFirst:   "vec4",
			minResults:  4,
			wantErr:     false,
		},
		{
			name:        "negative limit",
			query:       []float64{1, 1, 1},
			limit:       -1,
			wantErr:     true,
		},
		{
			name:        "nil query",
			query:       nil,
			limit:       1,
			wantErr:     true,
		},
		{
			name:        "empty query",
			query:       []float64{},
			limit:       1,
			wantErr:     true,
		},
		{
			name:        "wrong dimension",
			query:       []float64{1, 2},
			limit:       1,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := vs.Search(tt.query, tt.limit)
			if (err != nil) != tt.wantErr {
				t.Errorf("Search() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			if len(results) < tt.minResults {
				t.Errorf("got %d results, want at least %d", len(results), tt.minResults)
			}

			if tt.wantFirst != "" && len(results) > 0 && results[0].Key != tt.wantFirst {
				t.Errorf("got first result = %s, want %s", results[0].Key, tt.wantFirst)
			}

			// Verify results are sorted by similarity
			for i := 1; i < len(results); i++ {
				if results[i].Similarity > results[i-1].Similarity {
					t.Errorf("results not properly sorted at index %d", i)
				}
			}

			// Verify similarity scores are in valid range (with tolerance for floating point precision)
			for i, result := range results {
				if result.Similarity < -1.000001 || result.Similarity > 1.000001 {
					t.Errorf("result %d has invalid similarity score: %f", i, result.Similarity)
				}
			}
		})
	}
}

func TestVectorStoreUtilityFunctions(t *testing.T) {
	vs := New(DefaultLSHParams())
	
	t.Run("empty store", func(t *testing.T) {
		if vs.Size() != 0 {
			t.Errorf("Size() = %v, want 0", vs.Size())
		}
		
		keys := vs.Keys()
		if len(keys) != 0 {
			t.Errorf("Keys() = %v, want empty slice", keys)
		}
		
		if vs.Contains("nonexistent") {
			t.Error("Contains() should return false for empty store")
		}
	})
	
	// Add some vectors
	vectors := map[string][]float64{
		"vec1": {1, 2, 3},
		"vec2": {4, 5, 6},
		"vec3": {7, 8, 9},
	}
	
	for k, v := range vectors {
		if err := vs.Add(k, v); err != nil {
			t.Fatalf("failed to add vector %s: %v", k, err)
		}
	}
	
	t.Run("populated store", func(t *testing.T) {
		if vs.Size() != 3 {
			t.Errorf("Size() = %v, want 3", vs.Size())
		}
		
		keys := vs.Keys()
		if len(keys) != 3 {
			t.Errorf("Keys() length = %v, want 3", len(keys))
		}
		
		// Check all keys are present
		keyMap := make(map[string]bool)
		for _, key := range keys {
			keyMap[key] = true
		}
		for expectedKey := range vectors {
			if !keyMap[expectedKey] {
				t.Errorf("Keys() missing key %s", expectedKey)
			}
		}
		
		// Test Contains
		for key := range vectors {
			if !vs.Contains(key) {
				t.Errorf("Contains(%s) = false, want true", key)
			}
		}
		
		if vs.Contains("nonexistent") {
			t.Error("Contains() should return false for non-existent key")
		}
	})
	
	t.Run("clear store", func(t *testing.T) {
		vs.Clear()
		
		if vs.Size() != 0 {
			t.Errorf("Size() after Clear() = %v, want 0", vs.Size())
		}
		
		for key := range vectors {
			if vs.Contains(key) {
				t.Errorf("Contains(%s) = true after Clear(), want false", key)
			}
		}
		
		// Should be able to add new vectors after clear
		err := vs.Add("new", []float64{1, 2, 3})
		if err != nil {
			t.Errorf("Add() after Clear() failed: %v", err)
		}
	})
}

func TestVectorStoreStats(t *testing.T) {
	vs := New(DefaultLSHParams())
	
	// Test empty store stats
	stats := vs.Stats()
	if stats.VectorCount != 0 {
		t.Errorf("VectorCount = %v, want 0", stats.VectorCount)
	}
	if stats.Dimension != 0 {
		t.Errorf("Dimension = %v, want 0", stats.Dimension)
	}
	
	// Add vectors and test stats
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("vec%d", i)
		vector := []float64{float64(i), float64(i + 1), float64(i + 2)}
		if err := vs.Add(key, vector); err != nil {
			t.Fatalf("failed to add vector: %v", err)
		}
	}
	
	stats = vs.Stats()
	if stats.VectorCount != 10 {
		t.Errorf("VectorCount = %v, want 10", stats.VectorCount)
	}
	if stats.Dimension != 3 {
		t.Errorf("Dimension = %v, want 3", stats.Dimension)
	}
	if stats.LSHTables != DefaultLSHParams().NumHashTables {
		t.Errorf("LSHTables = %v, want %v", stats.LSHTables, DefaultLSHParams().NumHashTables)
	}
	if stats.TotalBuckets == 0 {
		t.Error("TotalBuckets should be > 0")
	}
}

func TestConcurrentOperations(t *testing.T) {
	vs := New(DefaultLSHParams())
	numOps := 1000
	var wg sync.WaitGroup

	// Concurrent additions
	for i := 0; i < numOps; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("key%d", i)
			vector := []float64{float64(i), float64(i), float64(i)}
			err := vs.Add(key, vector)
			if err != nil {
				t.Errorf("concurrent Add failed: %v", err)
			}
		}(i)
	}

	// Concurrent searches
	for i := 0; i < numOps/10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			query := []float64{1, 1, 1}
			_, err := vs.Search(query, 5)
			if err != nil {
				t.Errorf("concurrent Search failed: %v", err)
			}
		}()
	}

	// Concurrent deletions
	for i := 0; i < numOps/20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("key%d", i*20)
			vs.Delete(key)
		}(i)
	}

	// Concurrent batch operations
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			vectors := make(map[string][]float64)
			for j := 0; j < 10; j++ {
				key := fmt.Sprintf("batch%d_%d", i, j)
				vectors[key] = []float64{float64(i), float64(j), float64(i + j)}
			}
			err := vs.AddBatch(vectors)
			if err != nil {
				t.Errorf("concurrent AddBatch failed: %v", err)
			}
		}(i)
	}

	wg.Wait()

	// Verify store is still functional
	_, err := vs.Search([]float64{1, 1, 1}, 10)
	if err != nil {
		t.Errorf("final search failed: %v", err)
	}
}

func TestSearchStrategies(t *testing.T) {
	// Test sequential search (small number of tables)
	t.Run("sequential search", func(t *testing.T) {
		params := LSHParams{
			NumHashTables:    2, // Less than minGoroutineThreshold
			NumHashFunctions: 4,
			BucketWidth:     4.0,
		}
		vs := New(params)
		
		// Add test vectors
		for i := 0; i < 10; i++ {
			key := fmt.Sprintf("vec%d", i)
			vector := []float64{float64(i), float64(i + 1), float64(i + 2)}
			if err := vs.Add(key, vector); err != nil {
				t.Fatalf("failed to add vector: %v", err)
			}
		}
		
		results, err := vs.Search([]float64{5, 6, 7}, 3)
		if err != nil {
			t.Errorf("sequential search failed: %v", err)
		}
		if len(results) == 0 {
			t.Error("sequential search returned no results")
		}
	})
	
	// Test concurrent search (large number of tables)
	t.Run("concurrent search", func(t *testing.T) {
		params := LSHParams{
			NumHashTables:    10, // More than minGoroutineThreshold
			NumHashFunctions: 8,
			BucketWidth:     4.0,
		}
		vs := New(params)
		
		// Add test vectors
		for i := 0; i < 10; i++ {
			key := fmt.Sprintf("vec%d", i)
			vector := []float64{float64(i), float64(i + 1), float64(i + 2)}
			if err := vs.Add(key, vector); err != nil {
				t.Fatalf("failed to add vector: %v", err)
			}
		}
		
		results, err := vs.Search([]float64{5, 6, 7}, 3)
		if err != nil {
			t.Errorf("concurrent search failed: %v", err)
		}
		if len(results) == 0 {
			t.Error("concurrent search returned no results")
		}
	})
}

func TestLSHAccuracy(t *testing.T) {
	vs := New(DefaultLSHParams())
	dim := 128
	numVectors := 1000

	// Add random vectors
	vectors := make(map[string][]float64)
	for i := 0; i < numVectors; i++ {
		key := fmt.Sprintf("vec%d", i)
		vector := make([]float64, dim)
		for j := range vector {
			vector[j] = rand.NormFloat64()
		}
		vectors[key] = vector
		if err := vs.Add(key, vector); err != nil {
			t.Fatalf("failed to add vector: %v", err)
		}
	}

	// Test search accuracy
	numQueries := 10
	for i := 0; i < numQueries; i++ {
		query := make([]float64, dim)
		for j := range query {
			query[j] = rand.NormFloat64()
		}

		results, err := vs.Search(query, 10)
		if err != nil {
			t.Fatalf("search failed: %v", err)
		}

		if len(results) == 0 {
			t.Error("no results returned")
		}

		// Verify results are reasonable
		for _, result := range results {
			if result.Similarity < -1 || result.Similarity > 1 {
				t.Errorf("invalid similarity score: %f", result.Similarity)
			}
		}
		
		// Verify similarity calculation matches vectormath package
		if len(results) > 0 {
			key := results[0].Key
			vector := vectors[key]
			expectedSim, err := vectormath.CosineSimilarity(query, vector)
			if err != nil {
				t.Fatalf("vectormath similarity calculation failed: %v", err)
			}
			if !almostEqual(results[0].Similarity, expectedSim, 1e-10) {
				t.Errorf("similarity mismatch: got %f, want %f", results[0].Similarity, expectedSim)
			}
		}
	}
}

func TestErrorConditions(t *testing.T) {
	t.Run("search empty store", func(t *testing.T) {
		vs := New(DefaultLSHParams())
		_, err := vs.Search([]float64{1, 2, 3}, 5)
		if err != ErrStoreEmpty {
			t.Errorf("Search() on empty store error = %v, want %v", err, ErrStoreEmpty)
		}
	})
	
	t.Run("invalid LSH parameters", func(t *testing.T) {
		params := LSHParams{
			NumHashTables:    0,
			NumHashFunctions: 4,
			BucketWidth:     4.0,
		}
		vs := New(params)
		err := vs.Add("test", []float64{1, 2, 3})
		if err == nil {
			t.Error("Add() should fail with invalid LSH parameters")
		}
	})
}

// Benchmarks

func BenchmarkVectorStore(b *testing.B) {
	dim := 128
	vs := New(DefaultLSHParams())

	// Add some initial vectors
	numVectors := 10000
	for i := 0; i < numVectors; i++ {
		vector := make([]float64, dim)
		for j := range vector {
			vector[j] = rand.NormFloat64()
		}
		key := fmt.Sprintf("vec%d", i)
		if err := vs.Add(key, vector); err != nil {
			b.Fatalf("failed to add vector: %v", err)
		}
	}

	// Benchmark Add
	b.Run("Add", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			vector := make([]float64, dim)
			for j := range vector {
				vector[j] = rand.NormFloat64()
			}
			key := fmt.Sprintf("bench%d", i)
			_ = vs.Add(key, vector)
		}
	})

	// Benchmark AddBatch
	b.Run("AddBatch", func(b *testing.B) {
		batchSize := 100
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			vectors := make(map[string][]float64, batchSize)
			for j := 0; j < batchSize; j++ {
				vector := make([]float64, dim)
				for k := range vector {
					vector[k] = rand.NormFloat64()
				}
				key := fmt.Sprintf("batch%d_%d", i, j)
				vectors[key] = vector
			}
			_ = vs.AddBatch(vectors)
		}
	})

	// Benchmark Search
	query := make([]float64, dim)
	for i := range query {
		query[i] = rand.NormFloat64()
	}

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = vs.Search(query, 10)
		}
	})
	
	b.Run("SearchSequential", func(b *testing.B) {
		// Create store with few tables to force sequential search
		vsSeq := New(LSHParams{
			NumHashTables:    2,
			NumHashFunctions: 4,
			BucketWidth:     4.0,
		})
		
		// Add vectors
		for i := 0; i < 1000; i++ {
			vector := make([]float64, dim)
			for j := range vector {
				vector[j] = rand.NormFloat64()
			}
			key := fmt.Sprintf("vec%d", i)
			vsSeq.Add(key, vector)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = vsSeq.Search(query, 10)
		}
	})
	
	b.Run("SearchConcurrent", func(b *testing.B) {
		// Create store with many tables to force concurrent search
		vsCon := New(LSHParams{
			NumHashTables:    10,
			NumHashFunctions: 8,
			BucketWidth:     4.0,
		})
		
		// Add vectors
		for i := 0; i < 1000; i++ {
			vector := make([]float64, dim)
			for j := range vector {
				vector[j] = rand.NormFloat64()
			}
			key := fmt.Sprintf("vec%d", i)
			vsCon.Add(key, vector)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = vsCon.Search(query, 10)
		}
	})

	// Benchmark Delete
	b.Run("Delete", func(b *testing.B) {
		keys := make([]string, b.N)
		for i := 0; i < b.N; i++ {
			keys[i] = fmt.Sprintf("vec%d", i%numVectors)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vs.Delete(keys[i])
		}
	})
}

func BenchmarkMemoryAllocations(b *testing.B) {
	dim := 128
	vs := New(DefaultLSHParams())
	
	// Pre-populate
	for i := 0; i < 1000; i++ {
		vector := make([]float64, dim)
		for j := range vector {
			vector[j] = rand.NormFloat64()
		}
		vs.Add(fmt.Sprintf("vec%d", i), vector)
	}
	
	query := make([]float64, dim)
	for i := range query {
		query[i] = rand.NormFloat64()
	}
	
	b.Run("Search-Allocations", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = vs.Search(query, 10)
		}
	})
	
	b.Run("HashVector-Allocations", func(b *testing.B) {
		b.ReportAllocs()
		table := &vs.lshTables[0]
		for i := 0; i < b.N; i++ {
			_ = vs.hashVectorOptimized(table, query)
		}
	})
}

func BenchmarkConcurrentOperations(b *testing.B) {
	vs := New(DefaultLSHParams())
	dim := 128
	
	// Pre-populate
	for i := 0; i < 1000; i++ {
		vector := make([]float64, dim)
		for j := range vector {
			vector[j] = rand.NormFloat64()
		}
		vs.Add(fmt.Sprintf("vec%d", i), vector)
	}
	
	b.Run("ConcurrentSearch", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			query := make([]float64, dim)
			for i := range query {
				query[i] = rand.NormFloat64()
			}
			
			for pb.Next() {
				_, _ = vs.Search(query, 10)
			}
		})
	})
	
	b.Run("ConcurrentMixed", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				switch rand.Intn(4) {
				case 0: // Add
					vector := make([]float64, dim)
					for j := range vector {
						vector[j] = rand.NormFloat64()
					}
					vs.Add(fmt.Sprintf("bench%d", rand.Int()), vector)
				case 1: // Search
					query := make([]float64, dim)
					for j := range query {
						query[j] = rand.NormFloat64()
					}
					vs.Search(query, 5)
				case 2: // Delete
					vs.Delete(fmt.Sprintf("vec%d", rand.Intn(1000)))
				case 3: // Get
					vs.Get(fmt.Sprintf("vec%d", rand.Intn(1000)))
				}
			}
		})
	})
}

func BenchmarkDifferentDimensions(b *testing.B) {
	dimensions := []int{50, 128, 256, 512, 1024}
	
	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim-%d", dim), func(b *testing.B) {
			vs := New(DefaultLSHParams())
			
			// Add vectors
			for i := 0; i < 100; i++ {
				vector := make([]float64, dim)
				for j := range vector {
					vector[j] = rand.NormFloat64()
				}
				vs.Add(fmt.Sprintf("vec%d", i), vector)
			}
			
			query := make([]float64, dim)
			for i := range query {
				query[i] = rand.NormFloat64()
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = vs.Search(query, 10)
			}
		})
	}
}

// Helper functions

func vectorEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !almostEqual(a[i], b[i], 1e-6) {
			return false
		}
	}
	return true
}

func almostEqual(a, b float64, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

func init() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())
}