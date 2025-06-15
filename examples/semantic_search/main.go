package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/colesmcintosh/ai-memory/internal/vectorstore"
)

const (
	maxRetries     = 3
	retryDelay     = time.Second
	batchSize      = 10
	embeddingModel = openai.AdaEmbeddingV2
)

// Document represents a text document with its embedding
type Document struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding,omitempty"` // Omit for smaller JSON files
}

// Store represents the persistent storage for documents and their embeddings
type Store struct {
	vs        *vectorstore.Store
	documents map[string]Document
	client    *openai.Client
	dataPath  string
	mu        sync.RWMutex // Protect concurrent access
}

// NewStore creates a new store with optimized LSH parameters
func NewStore(apiKey, dataPath string) (*Store, error) {
	// Optimized LSH parameters for semantic search
	lshParams := vectorstore.LSHParams{
		NumHashTables:    8,  // More tables for better recall
		NumHashFunctions: 12, // More functions for better precision
		BucketWidth:     3.0, // Tighter buckets for semantic similarity
	}
	
	s := &Store{
		vs:        vectorstore.New(lshParams),
		documents: make(map[string]Document),
		client:    openai.NewClient(apiKey),
		dataPath:  dataPath,
	}

	return s, s.load()
}

// getEmbeddingWithRetry gets embedding with exponential backoff retry
func (s *Store) getEmbeddingWithRetry(ctx context.Context, text string) ([]float64, error) {
	var lastErr error
	
	for attempt := 0; attempt < maxRetries; attempt++ {
		embedding, err := s.getEmbedding(ctx, text)
		if err == nil {
			return embedding, nil
		}
		
		lastErr = err
		if attempt < maxRetries-1 {
			time.Sleep(retryDelay * time.Duration(1<<attempt)) // Exponential backoff
		}
	}
	
	return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, lastErr)
}

// getEmbedding gets the embedding for a text using OpenAI's API
func (s *Store) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	resp, err := s.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: embeddingModel,
		Input: []string{text},
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data received")
	}

	// Convert []float32 to []float64 efficiently
	src := resp.Data[0].Embedding
	dst := make([]float64, len(src))
	for i, v := range src {
		dst[i] = float64(v)
	}

	return dst, nil
}

// getBatchEmbeddings gets embeddings for multiple texts efficiently
func (s *Store) getBatchEmbeddings(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	resp, err := s.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: embeddingModel,
		Input: texts,
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(resp.Data))
	}

	embeddings := make([][]float64, len(texts))
	for i, data := range resp.Data {
		embeddings[i] = make([]float64, len(data.Embedding))
		for j, v := range data.Embedding {
			embeddings[i][j] = float64(v)
		}
	}

	return embeddings, nil
}

// AddDocument adds a new document to the store
func (s *Store) AddDocument(ctx context.Context, id, text string) error {
	embedding, err := s.getEmbeddingWithRetry(ctx, text)
	if err != nil {
		return fmt.Errorf("failed to get embedding: %w", err)
	}

	return s.addDocumentWithEmbedding(id, text, embedding)
}

// AddDocumentsBatch adds multiple documents efficiently using batch API calls
func (s *Store) AddDocumentsBatch(ctx context.Context, docs map[string]string) error {
	if len(docs) == 0 {
		return nil
	}

	// Process in batches to respect API limits
	ids := make([]string, 0, len(docs))
	texts := make([]string, 0, len(docs))
	
	for id, text := range docs {
		ids = append(ids, id)
		texts = append(texts, text)
	}

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batchTexts := texts[i:end]
		batchIDs := ids[i:end]

		embeddings, err := s.getBatchEmbeddings(ctx, batchTexts)
		if err != nil {
			return fmt.Errorf("failed to get batch embeddings: %w", err)
		}

		// Add to store
		for j, embedding := range embeddings {
			if err := s.addDocumentWithEmbedding(batchIDs[j], batchTexts[j], embedding); err != nil {
				return fmt.Errorf("failed to add document %s: %w", batchIDs[j], err)
			}
		}
	}

	return s.save()
}

// addDocumentWithEmbedding adds a document with pre-computed embedding
func (s *Store) addDocumentWithEmbedding(id, text string, embedding []float64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.vs.Add(id, embedding); err != nil {
		return fmt.Errorf("failed to add to vector store: %w", err)
	}

	s.documents[id] = Document{
		ID:        id,
		Text:      text,
		Embedding: embedding,
	}

	return nil
}

// Search finds similar documents to the query text
func (s *Store) Search(ctx context.Context, query string, limit int) ([]Document, error) {
	embedding, err := s.getEmbeddingWithRetry(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %w", err)
	}

	results, err := s.vs.Search(embedding, limit)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	s.mu.RLock()
	docs := make([]Document, 0, len(results))
	for _, result := range results {
		if doc, ok := s.documents[result.Key]; ok {
			docs = append(docs, doc)
		}
	}
	s.mu.RUnlock()

	return docs, nil
}

// save persists the current state to disk efficiently
func (s *Store) save() error {
	s.mu.RLock()
	data := struct {
		Documents map[string]Document `json:"documents"`
	}{Documents: s.documents}
	s.mu.RUnlock()

	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(s.dataPath), 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %w", err)
	}

	// Write to temporary file first, then rename for atomic operation
	tempPath := s.dataPath + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to encode data: %w", err)
	}

	file.Close()
	return os.Rename(tempPath, s.dataPath)
}

// load restores the state from disk
func (s *Store) load() error {
	file, err := os.Open(s.dataPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to open data file: %w", err)
	}
	defer file.Close()

	var data struct {
		Documents map[string]Document `json:"documents"`
	}

	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return fmt.Errorf("failed to decode data: %w", err)
	}

	// Restore documents and their embeddings
	for _, doc := range data.Documents {
		if err := s.vs.Add(doc.ID, doc.Embedding); err != nil {
			return fmt.Errorf("failed to restore vector %s: %w", doc.ID, err)
		}
		s.documents[doc.ID] = doc
	}

	return nil
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	store, err := NewStore(apiKey, "data/semantic_search.json")
	if err != nil {
		log.Fatalf("Failed to create store: %v", err)
	}

	ctx := context.Background()

	// Add example documents efficiently using batch processing
	if len(store.documents) == 0 {
		docs := map[string]string{
			"go":         "Go is an open source programming language that makes it easy to build simple, reliable, and efficient software.",
			"python":     "Python is a programming language that lets you work quickly and integrate systems more effectively.",
			"rust":       "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
			"javascript": "JavaScript is a lightweight, interpreted programming language designed for creating network-centric applications.",
			"swift":      "Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, tvOS, and watchOS.",
		}

		fmt.Println("Adding example documents using batch processing...")
		start := time.Now()
		
		if err := store.AddDocumentsBatch(ctx, docs); err != nil {
			log.Fatalf("Failed to add documents: %v", err)
		}
		
		fmt.Printf("Added %d documents in %v\n", len(docs), time.Since(start))
	}

	// Perform semantic search
	queries := []string{
		"fast systems programming languages",
		"web development technologies", 
		"mobile app development",
	}

	for _, query := range queries {
		fmt.Printf("\nQuery: %s\n", query)
		
		start := time.Now()
		results, err := store.Search(ctx, query, 3)
		if err != nil {
			log.Printf("Search failed for '%s': %v", query, err)
			continue
		}
		
		fmt.Printf("Search completed in %v\n", time.Since(start))
		fmt.Println("Most semantically similar documents:")
		
		for i, doc := range results {
			fmt.Printf("%d. [%s] %s\n", i+1, doc.ID, doc.Text)
		}
	}
} 