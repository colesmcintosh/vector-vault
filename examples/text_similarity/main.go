package main

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/colesmcintosh/ai-memory/internal/vectorstore"
	"github.com/colesmcintosh/ai-memory/pkg/vectormath"
)

const (
	vectorDim = 256 // Increased dimension for better representation
	topK      = 3
)

// TextProcessor handles efficient text processing and embedding generation
type TextProcessor struct {
	vocabulary map[string]int // Word to index mapping
	idf        map[string]float64 // Inverse document frequency scores
	docCount   int
}

// NewTextProcessor creates a new text processor
func NewTextProcessor() *TextProcessor {
	return &TextProcessor{
		vocabulary: make(map[string]int),
		idf:        make(map[string]float64),
	}
}

// preprocessText cleans and tokenizes text efficiently
func (tp *TextProcessor) preprocessText(text string) []string {
	// Convert to lowercase and split by whitespace
	text = strings.ToLower(text)
	
	// Simple tokenization - remove punctuation and split
	words := strings.FieldsFunc(text, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	
	// Filter out very short words
	filtered := words[:0]
	for _, word := range words {
		if len(word) > 2 { // Keep words longer than 2 characters
			filtered = append(filtered, word)
		}
	}
	
	return filtered
}

// buildVocabulary builds vocabulary and calculates IDF scores from a corpus
func (tp *TextProcessor) buildVocabulary(texts []string) {
	tp.docCount = len(texts)
	wordDocCount := make(map[string]int)
	
	// Count documents containing each word
	for _, text := range texts {
		words := tp.preprocessText(text)
		seen := make(map[string]bool)
		
		for _, word := range words {
			if !seen[word] {
				wordDocCount[word]++
				seen[word] = true
			}
		}
	}
	
	// Build vocabulary and calculate IDF
	idx := 0
	for word, docCount := range wordDocCount {
		tp.vocabulary[word] = idx
		tp.idf[word] = math.Log(float64(tp.docCount) / float64(docCount))
		idx++
	}
	
	// Ensure vocabulary fits in vector dimension
	if len(tp.vocabulary) > vectorDim {
		// Keep only the most informative words (highest IDF)
		type wordIDF struct {
			word string
			idf  float64
		}
		
		wordIDFs := make([]wordIDF, 0, len(tp.vocabulary))
		for word, idfVal := range tp.idf {
			wordIDFs = append(wordIDFs, wordIDF{word, idfVal})
		}
		
		// Sort by IDF descending
		sort.Slice(wordIDFs, func(i, j int) bool {
			return wordIDFs[i].idf > wordIDFs[j].idf
		})
		
		// Keep top words
		tp.vocabulary = make(map[string]int)
		tp.idf = make(map[string]float64)
		
		for i, wi := range wordIDFs[:vectorDim] {
			tp.vocabulary[wi.word] = i
			tp.idf[wi.word] = wi.idf
		}
	}
}

// generateEmbedding creates a TF-IDF-like embedding for text
func (tp *TextProcessor) generateEmbedding(text string) []float64 {
	vector := make([]float64, vectorDim)
	words := tp.preprocessText(text)
	
	// Calculate term frequencies
	tf := make(map[string]int)
	for _, word := range words {
		tf[word]++
	}
	
	// Generate TF-IDF vector
	totalWords := len(words)
	if totalWords == 0 {
		return vector
	}
	
	// Fill vector with TF-IDF scores
	for word, freq := range tf {
		if idx, exists := tp.vocabulary[word]; exists {
			// TF-IDF = (term_freq / total_words) * idf
			tfScore := float64(freq) / float64(totalWords)
			vector[idx] = tfScore * tp.idf[word]
		}
	}
	
	// Add n-gram features for better representation
	tp.addNGramFeatures(vector, words)
	
	// Normalize the vector
	if err := vectormath.Normalize(vector); err != nil {
		log.Printf("Failed to normalize vector: %v", err)
	}
	
	return vector
}

// addNGramFeatures adds bigram features to improve text representation
func (tp *TextProcessor) addNGramFeatures(vector []float64, words []string) {
	if len(words) < 2 {
		return
	}
	
	// Add bigram features using hash-based indexing
	for i := 0; i < len(words)-1; i++ {
		bigram := words[i] + "_" + words[i+1]
		// Use simple hash to map bigram to vector position
		hash := tp.simpleHash(bigram) % (vectorDim / 2) // Use second half for bigrams
		vector[vectorDim/2 + hash] += 0.5 // Lower weight for bigrams
	}
}

// simpleHash implements a simple hash function
func (tp *TextProcessor) simpleHash(s string) int {
	hash := 0
	for _, c := range s {
		hash = hash*31 + int(c)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}

// DocumentStore manages documents and their embeddings efficiently
type DocumentStore struct {
	vs        *vectorstore.Store
	docs      []Document
	processor *TextProcessor
}

// Document represents a document with its metadata
type Document struct {
	Key  string
	Text string
}

// NewDocumentStore creates a new document store
func NewDocumentStore(documents []Document) *DocumentStore {
	// Optimized LSH parameters for text similarity
	lshParams := vectorstore.LSHParams{
		NumHashTables:    10, // More tables for better recall
		NumHashFunctions: 16, // More functions for better precision  
		BucketWidth:     2.5, // Tighter buckets for text similarity
	}
	
	processor := NewTextProcessor()
	
	// Build vocabulary from all documents
	texts := make([]string, len(documents))
	for i, doc := range documents {
		texts[i] = doc.Text
	}
	processor.buildVocabulary(texts)
	
	ds := &DocumentStore{
		vs:        vectorstore.New(lshParams),
		docs:      documents,
		processor: processor,
	}
	
	// Add all documents to vector store
	for _, doc := range documents {
		embedding := processor.generateEmbedding(doc.Text)
		if err := ds.vs.Add(doc.Key, embedding); err != nil {
			log.Printf("Failed to add document %s: %v", doc.Key, err)
		}
	}
	
	return ds
}

// Search performs similarity search
func (ds *DocumentStore) Search(query string, k int) ([]Result, error) {
	queryEmbedding := ds.processor.generateEmbedding(query)
	
	results, err := ds.vs.Search(queryEmbedding, k)
	if err != nil {
		return nil, err
	}
	
	// Convert to our result format with original text
	searchResults := make([]Result, len(results))
	for i, result := range results {
		var text string
		for _, doc := range ds.docs {
			if doc.Key == result.Key {
				text = doc.Text
				break
			}
		}
		searchResults[i] = Result{
			Key:        result.Key,
			Text:       text,
			Similarity: result.Similarity,
		}
	}
	
	return searchResults, nil
}

// Result represents a search result
type Result struct {
	Key        string
	Text       string
	Similarity float64
}

func main() {
	// Enhanced sample texts with more variety
	documents := []Document{
		{"doc1", "The quick brown fox jumps over the lazy dog"},
		{"doc2", "A lazy dog sleeps peacefully in the warm sun"},
		{"doc3", "Quick foxes are known for their incredible jumping ability"},
		{"doc4", "The weather is sunny and warm today"},
		{"doc5", "Dogs and cats are popular domestic pets"},
		{"doc6", "Programming languages like Go and Python are powerful"},
		{"doc7", "Machine learning algorithms process large datasets"},
		{"doc8", "The fox ran quickly through the forest"},
		{"doc9", "Sleeping dogs lie quietly in comfortable places"},
		{"doc10", "Sunny weather brings happiness to many people"},
	}

	fmt.Printf("Creating document store with %d documents...\n", len(documents))
	
	start := time.Now()
	ds := NewDocumentStore(documents)
	indexTime := time.Since(start)
	
	fmt.Printf("Indexing completed in: %v\n", indexTime)
	fmt.Printf("Vector store size: %d documents\n", ds.vs.Size())

	// Test queries with different similarity patterns
	queries := []string{
		"A fox jumping quickly",
		"Dog sleeping in sunshine", 
		"Programming and algorithms",
		"Fast animals in nature",
	}

	fmt.Println("\nPerforming similarity searches...")
	
	for _, query := range queries {
		fmt.Printf("\n" + strings.Repeat("=", 50) + "\n")
		fmt.Printf("Query: %s\n", query)
		
		start := time.Now()
		results, err := ds.Search(query, topK)
		searchTime := time.Since(start)
		
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		fmt.Printf("Search completed in: %v\n", searchTime)
		fmt.Println("\nMost similar documents:")
		
		for i, result := range results {
			fmt.Printf("%d. %s (similarity: %.4f)\n   Text: %s\n", 
				i+1, result.Key, result.Similarity, result.Text)
		}
	}
} 