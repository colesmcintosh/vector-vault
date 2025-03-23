package vectorstore

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
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
			}
		})
	}
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
	vs.Delete(key)

	// Verify deletion
	_, ok := vs.Get(key)
	if ok {
		t.Error("vector still exists after deletion")
	}

	// Delete non-existent key (should not panic)
	vs.Delete("nonexistent")
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
	}{
		{
			name:        "exact match x-axis",
			query:       []float64{1, 0, 0},
			limit:       1,
			wantFirst:   "vec1",
			minResults:  1,
		},
		{
			name:        "similar to diagonal",
			query:       []float64{1, 1, 0.9},
			limit:       2,
			wantFirst:   "vec4",
			minResults:  2,
		},
		{
			name:        "all results",
			query:       []float64{1, 1, 1},
			limit:       0,
			wantFirst:   "vec4",
			minResults:  4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := vs.Search(tt.query, tt.limit)
			if err != nil {
				t.Fatalf("Search() error = %v", err)
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
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name    string
		vec1    []float64
		vec2    []float64
		want    float64
		wantErr bool
	}{
		{
			name:    "identical vectors",
			vec1:    []float64{1, 2, 3},
			vec2:    []float64{1, 2, 3},
			want:    1.0,
			wantErr: false,
		},
		{
			name:    "orthogonal vectors",
			vec1:    []float64{1, 0, 0},
			vec2:    []float64{0, 1, 0},
			want:    0.0,
			wantErr: false,
		},
		{
			name:    "opposite vectors",
			vec1:    []float64{1, 2, 3},
			vec2:    []float64{-1, -2, -3},
			want:    -1.0,
			wantErr: false,
		},
		{
			name:    "different dimensions",
			vec1:    []float64{1, 2},
			vec2:    []float64{1, 2, 3},
			want:    0,
			wantErr: true,
		},
		{
			name:    "zero vector",
			vec1:    []float64{0, 0, 0},
			vec2:    []float64{1, 2, 3},
			want:    0,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := CosineSimilarity(tt.vec1, tt.vec2)
			if (err != nil) != tt.wantErr {
				t.Errorf("CosineSimilarity() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !almostEqual(got, tt.want, 1e-6) {
				t.Errorf("CosineSimilarity() = %v, want %v", got, tt.want)
			}
		})
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

	wg.Wait()
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
	}
}

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
}

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
}