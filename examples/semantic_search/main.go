package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/sashabaranov/go-openai"
	"github.com/colesmcintosh/ai-memory/internal/vectorstore"
)

// Document represents a text document with its embedding
type Document struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding"`
}

// Store represents the persistent storage for documents and their embeddings
type Store struct {
	vs        *vectorstore.Store
	documents map[string]Document
	client    *openai.Client
	dataPath  string
}

// NewStore creates a new store with the given OpenAI API key and data path
func NewStore(apiKey, dataPath string) (*Store, error) {
	client := openai.NewClient(apiKey)
	
	// Create a new vector store with default LSH parameters
	vs := vectorstore.New(vectorstore.DefaultLSHParams())
	
	s := &Store{
		vs:        vs,
		documents: make(map[string]Document),
		client:    client,
		dataPath:  dataPath,
	}

	// Load existing data if available
	if err := s.load(); err != nil {
		return nil, fmt.Errorf("failed to load data: %v", err)
	}

	return s, nil
}

// getEmbedding gets the embedding for a text using OpenAI's API
func (s *Store) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	resp, err := s.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.AdaEmbeddingV2,
		Input: []string{text},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding: %v", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data received")
	}

	// Convert []float32 to []float64
	embedding := make([]float64, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float64(v)
	}

	return embedding, nil
}

// AddDocument adds a new document to the store
func (s *Store) AddDocument(ctx context.Context, id, text string) error {
	embedding, err := s.getEmbedding(ctx, text)
	if err != nil {
		return err
	}

	doc := Document{
		ID:        id,
		Text:      text,
		Embedding: embedding,
	}

	if err := s.vs.Add(id, embedding); err != nil {
		return fmt.Errorf("failed to add to vector store: %v", err)
	}

	s.documents[id] = doc
	return s.save()
}

// Search finds similar documents to the query text
func (s *Store) Search(ctx context.Context, query string, limit int) ([]Document, error) {
	embedding, err := s.getEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}

	results, err := s.vs.Search(embedding, limit)
	if err != nil {
		return nil, fmt.Errorf("search failed: %v", err)
	}

	docs := make([]Document, 0, len(results))
	for _, result := range results {
		if doc, ok := s.documents[result.Key]; ok {
			docs = append(docs, doc)
		}
	}

	return docs, nil
}

// save persists the current state to disk
func (s *Store) save() error {
	data := struct {
		Documents map[string]Document `json:"documents"`
	}{
		Documents: s.documents,
	}

	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %v", err)
	}

	if err := os.MkdirAll(filepath.Dir(s.dataPath), 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	if err := os.WriteFile(s.dataPath, bytes, 0644); err != nil {
		return fmt.Errorf("failed to write data file: %v", err)
	}

	return nil
}

// load restores the state from disk
func (s *Store) load() error {
	bytes, err := os.ReadFile(s.dataPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read data file: %v", err)
	}

	var data struct {
		Documents map[string]Document `json:"documents"`
	}

	if err := json.Unmarshal(bytes, &data); err != nil {
		return fmt.Errorf("failed to unmarshal data: %v", err)
	}

	// Restore documents and their embeddings
	for _, doc := range data.Documents {
		if err := s.vs.Add(doc.ID, doc.Embedding); err != nil {
			return fmt.Errorf("failed to restore vector: %v", err)
		}
		s.documents[doc.ID] = doc
	}

	return nil
}

func main() {
	// Get OpenAI API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create a new store
	store, err := NewStore(apiKey, "data/semantic_search.json")
	if err != nil {
		log.Fatalf("Failed to create store: %v", err)
	}

	ctx := context.Background()

	// Add some example documents if the store is empty
	if len(store.documents) == 0 {
		documents := []struct {
			id   string
			text string
		}{
			{"1", "Go is an open source programming language that makes it easy to build simple, reliable, and efficient software."},
			{"2", "Python is a programming language that lets you work quickly and integrate systems more effectively."},
			{"3", "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety."},
			{"4", "JavaScript is a lightweight, interpreted programming language designed for creating network-centric applications."},
			{"5", "Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, tvOS, and watchOS."},
		}

		fmt.Println("Adding example documents...")
		for _, doc := range documents {
			if err := store.AddDocument(ctx, doc.id, doc.text); err != nil {
				log.Printf("Failed to add document %s: %v", doc.id, err)
				continue
			}
			fmt.Printf("Added document: %s\n", doc.text)
		}
	}

	// Perform a semantic search
	fmt.Println("\nPerforming semantic search...")
	query := "fast systems programming languages"
	results, err := store.Search(ctx, query, 3)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nQuery: %s\n", query)
	fmt.Println("\nMost semantically similar documents:")
	for i, doc := range results {
		fmt.Printf("%d. %s\n", i+1, doc.Text)
	}
} 