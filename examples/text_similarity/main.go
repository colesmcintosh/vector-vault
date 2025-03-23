package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/colesmcintosh/ai-memory/internal/vectorstore"
)

// Mock embedding function - in a real application, this would call an embedding API
func mockEmbedding(text string) []float64 {
	// This is a very simple mock that creates a vector based on word count frequencies
	// In a real application, you would use a proper embedding model
	words := strings.Fields(strings.ToLower(text))
	vector := make([]float64, 128) // Using 128 dimensions for this example
	
	// Create a simple frequency-based embedding
	for _, word := range words {
		// Use the first character of each word to determine the position
		pos := int(word[0]) % len(vector)
		vector[pos] += 1.0
	}
	
	// Normalize the vector
	var sum float64
	for _, v := range vector {
		sum += v * v
	}
	if sum > 0 {
		norm := float64(1.0 / float64(sum))
		for i := range vector {
			vector[i] *= norm
		}
	}
	
	return vector
}

func main() {
	// Create a new vector store
	vs := vectorstore.New(vectorstore.DefaultLSHParams())

	// Sample texts
	texts := []struct {
		key  string
		text string
	}{
		{"doc1", "The quick brown fox jumps over the lazy dog"},
		{"doc2", "A lazy dog sleeps in the sun"},
		{"doc3", "Quick foxes are known for jumping"},
		{"doc4", "The weather is sunny today"},
		{"doc5", "Dogs and cats are popular pets"},
	}

	// Add document embeddings to the vector store
	fmt.Println("Adding documents to the vector store...")
	for _, doc := range texts {
		embedding := mockEmbedding(doc.text)
		if err := vs.Add(doc.key, embedding); err != nil {
			log.Printf("Failed to add document %s: %v", doc.key, err)
			continue
		}
		fmt.Printf("Added document: %s\n", doc.text)
	}

	// Perform similarity search
	fmt.Println("\nPerforming similarity search...")
	queryText := "A fox jumping quickly"
	queryEmbedding := mockEmbedding(queryText)

	results, err := vs.Search(queryEmbedding, 3) // Get top 3 similar documents
	if err != nil {
		log.Printf("Search failed: %v", err)
		return
	}

	fmt.Printf("\nQuery: %s\n", queryText)
	fmt.Println("\nMost similar documents:")
	for _, result := range results {
		// Find the original text for the result
		var text string
		for _, doc := range texts {
			if doc.key == result.Key {
				text = doc.text
				break
			}
		}
		fmt.Printf("%s (similarity: %.4f): %s\n", result.Key, result.Similarity, text)
	}
} 