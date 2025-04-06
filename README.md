# VectorVault

A high-performance vector similarity search engine with LSH (Locality-Sensitive Hashing) optimization, written in Go.

## Overview

VectorVault is designed to be your secure and efficient vault for vector embeddings, providing:

- **Security**: Thread-safe operations and data integrity
- **Speed**: LSH-based similarity search with SIMD optimizations
- **Scalability**: Efficient memory usage and parallel processing
- **Simplicity**: Clean API design following Go idioms

## Features

- Fast similarity search using LSH (Locality-Sensitive Hashing)
- SIMD-optimized vector operations for maximum performance
- Thread-safe concurrent operations with mutex protection
- Parallel processing for search operations
- Configurable LSH parameters for fine-tuning
- Memory-efficient storage and retrieval
- Comprehensive test coverage and benchmarks

## Getting Started

### Prerequisites

- Go 1.21 or later
- Make (optional, for running commands)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/colesmcintosh/vectorvault.git
cd vectorvault
```

2. Install dependencies:
```bash
go mod tidy
```

### Quick Start Example

```go
package main

import (
    "fmt"
    "github.com/colesmcintosh/vectorvault/internal/vectorstore"
)

func main() {
    // Create a new vector store with default LSH parameters
    vs := vectorstore.New(vectorstore.DefaultLSHParams())

    // Add some vectors
    vs.Add("vec1", []float64{1, 2, 3})
    vs.Add("vec2", []float64{4, 5, 6})

    // Search for similar vectors
    results, _ := vs.Search([]float64{1, 2, 3}, 10)
    for _, result := range results {
        fmt.Printf("%s: similarity %.4f\n", result.Key, result.Similarity)
    }
}
```

## Documentation

### Project Structure

```
.
├── cmd/
│   └── vectorstore/        # Performance benchmark example
│       └── main.go
├── examples/
│   ├── text_similarity/    # Basic text similarity example
│   └── semantic_search/    # Advanced semantic search with OpenAI
├── internal/
│   └── vectorstore/        # Core vector store implementation
│       ├── store.go
│       └── store_test.go
├── pkg/
│   └── vectormath/         # Public vector math utilities
└── README.md
```

### Configuration

#### LSH Parameters

Configure VectorVault with custom LSH parameters:

```go
params := vectorstore.LSHParams{
    NumHashTables:    6,    // More tables = better recall, more memory
    NumHashFunctions: 8,    // More functions = better precision, slower
    BucketWidth:     4.0,  // Larger width = more matches, less precision
}
vs := vectorstore.New(params)
```

#### Thread Safety

All operations are thread-safe by default:

```go
// These operations can be performed concurrently
go func() { vs.Add("key1", vector1) }()
go func() { vs.Search(queryVector, 10) }()
go func() { vs.Delete("key2") }()
```

### Examples

1. Basic performance test:
```bash
go run cmd/vectorstore/main.go
```

2. Text similarity example:
```bash
go run examples/text_similarity/main.go
```

3. Semantic search example (requires OpenAI API key):
```bash
export OPENAI_API_KEY='your-api-key-here'
go run examples/semantic_search/main.go
```

## Development

### Running Tests

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./internal/vectorstore

# Run specific test
go test -v -run TestVectorStoreSearch ./internal/vectorstore
```

### Performance Metrics

VectorVault is optimized for both speed and memory efficiency:

- Vector addition: ~3.8µs per operation
- Memory efficient: ~1.8KB per vector
- Parallel search processing
- SIMD-optimized similarity calculations

#### Benchmark Results

```
BenchmarkVectorStore/Add-10    	  300000	      3824 ns/op
BenchmarkVectorStore/Search-10 	   50000	     31245 ns/op
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes using conventional commit format
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(search): add parallel processing for search operations

- Implements parallel processing for similarity calculations
- Adds worker pool for better resource management
- Updates documentation with new performance metrics

Closes #123
```

## License

MIT License - see LICENSE file for details 