package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/colesmcintosh/ai-memory/internal/vectorstore"
)

func generateRandomVector(dim int) []float64 {
	vector := make([]float64, dim)
	for i := range vector {
		vector[i] = rand.NormFloat64()
	}
	return vector
}

func main() {
	// Set random seed
	rand.Seed(time.Now().UnixNano())

	// Create a new vector store with custom LSH parameters
	vs := vectorstore.New(vectorstore.LSHParams{
		NumHashTables:    6,    // More hash tables for better recall
		NumHashFunctions: 8,    // More hash functions for better precision
		BucketWidth:     4.0,  // Bucket width affects similarity threshold
	})

	// Generate a large number of random vectors
	numVectors := 100000
	dimension := 128
	fmt.Printf("Generating %d random vectors of dimension %d...\n", numVectors, dimension)

	start := time.Now()
	for i := 0; i < numVectors; i++ {
		key := fmt.Sprintf("vector%d", i)
		vector := generateRandomVector(dimension)
		if err := vs.Add(key, vector); err != nil {
			log.Printf("Failed to add %s: %v", key, err)
			continue
		}
	}
	fmt.Printf("Vector generation and indexing took: %v\n", time.Since(start))

	// Perform similarity search
	fmt.Println("\nPerforming similarity searches...")
	
	// Generate a random query vector
	queryVector := generateRandomVector(dimension)
	
	// Time the search operation
	start = time.Now()
	results, err := vs.Search(queryVector, 10) // Get top 10 results
	if err != nil {
		log.Printf("Search failed: %v", err)
		return
	}
	searchTime := time.Since(start)

	fmt.Printf("\nSearch completed in: %v\n", searchTime)
	fmt.Println("\nTop matches:")
	for _, result := range results {
		fmt.Printf("%s: similarity %.4f\n", result.Key, result.Similarity)
	}

	// Store statistics
	fmt.Printf("\nStore size: %d vectors\n", vs.Size())
} 