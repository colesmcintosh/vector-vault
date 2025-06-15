package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/colesmcintosh/ai-memory/internal/vectorstore"
)

const (
	numVectors  = 100000
	dimension   = 128
	topK        = 10
	numWorkers  = 8 // Parallel vector generation workers
)

// vectorPool reuses vector slices to reduce allocations
var vectorPool = sync.Pool{
	New: func() interface{} {
		return make([]float64, dimension)
	},
}

// generateVectorBatch generates vectors in parallel batches
func generateVectorBatch(vs *vectorstore.Store, start, count int, wg *sync.WaitGroup, errChan chan<- error) {
	defer wg.Done()
	
	// Create local random source for thread safety
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))
	
	for i := 0; i < count; i++ {
		vector := vectorPool.Get().([]float64)
		
		// Generate random normal vector
		for j := range vector {
			vector[j] = rng.NormFloat64()
		}
		
		key := fmt.Sprintf("vector%d", start+i)
		if err := vs.Add(key, vector); err != nil {
			errChan <- fmt.Errorf("failed to add %s: %w", key, err)
			vectorPool.Put(vector)
			return
		}
		
		// Return vector to pool (vs.Add copies it internally)
		vectorPool.Put(vector)
	}
}

// timeOperation executes a function and returns its duration
func timeOperation(name string, fn func() error) error {
	start := time.Now()
	err := fn()
	fmt.Printf("%s took: %v\n", name, time.Since(start))
	return err
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	// Initialize vector store with optimized parameters
	vs := vectorstore.New(vectorstore.LSHParams{
		NumHashTables:    6,
		NumHashFunctions: 8,
		BucketWidth:     4.0,
	})

	fmt.Printf("Generating %d vectors (dim=%d) using %d workers...\n", numVectors, dimension, numWorkers)

	// Generate vectors in parallel
	err := timeOperation("Vector generation and indexing", func() error {
		var wg sync.WaitGroup
		errChan := make(chan error, numWorkers)
		
		batchSize := numVectors / numWorkers
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			start := i * batchSize
			count := batchSize
			
			// Handle remainder in last batch
			if i == numWorkers-1 {
				count = numVectors - start
			}
			
			go generateVectorBatch(vs, start, count, &wg, errChan)
		}
		
		go func() {
			wg.Wait()
			close(errChan)
		}()
		
		// Check for errors
		for err := range errChan {
			if err != nil {
				return err
			}
		}
		return nil
	})
	
	if err != nil {
		log.Fatal(err)
	}

	// Perform similarity search
	fmt.Println("\nPerforming similarity search...")
	
	var results []vectorstore.SearchResult
	err = timeOperation("Search", func() error {
		// Generate query vector
		queryVector := make([]float64, dimension)
		for i := range queryVector {
			queryVector[i] = rand.NormFloat64()
		}
		
		var err error
		results, err = vs.Search(queryVector, topK)
		return err
	})
	
	if err != nil {
		log.Fatal(err)
	}

	// Display results
	fmt.Printf("\nTop %d matches from %d vectors:\n", len(results), vs.Size())
	for i, result := range results {
		fmt.Printf("%2d. %s: %.4f\n", i+1, result.Key, result.Similarity)
	}
} 