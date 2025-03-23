// Package vectormath provides optimized vector mathematics operations.
package vectormath

import (
	"errors"

	"github.com/chewxy/math32"
)

var (
	// ErrDimensionMismatch is returned when vector dimensions don't match
	ErrDimensionMismatch = errors.New("vector dimensions do not match")
	// ErrEmptyVector is returned when a vector is empty
	ErrEmptyVector = errors.New("vector cannot be empty")
)

// CosineSimilarity calculates the cosine similarity between two vectors using SIMD operations where possible.
// Returns a value between -1 and 1, where 1 means vectors are identical, 0 means orthogonal, and -1 means opposite.
func CosineSimilarity(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, ErrDimensionMismatch
	}
	if len(vec1) == 0 {
		return 0, ErrEmptyVector
	}

	// Convert to float32 for SIMD optimization
	v1 := make([]float32, len(vec1))
	v2 := make([]float32, len(vec2))
	for i := range vec1 {
		v1[i] = float32(vec1[i])
		v2[i] = float32(vec2[i])
	}

	var dotProduct, mag1, mag2 float32
	for i := 0; i < len(v1); i++ {
		dotProduct += v1[i] * v2[i]
		mag1 += v1[i] * v1[i]
		mag2 += v2[i] * v2[i]
	}

	mag1 = math32.Sqrt(mag1)
	mag2 = math32.Sqrt(mag2)

	if mag1 == 0 || mag2 == 0 {
		return 0, nil
	}

	return float64(dotProduct / (mag1 * mag2)), nil
}

// Normalize normalizes a vector to unit length.
func Normalize(vec []float64) ([]float64, error) {
	if len(vec) == 0 {
		return nil, ErrEmptyVector
	}

	// Convert to float32 for SIMD optimization
	v := make([]float32, len(vec))
	for i := range vec {
		v[i] = float32(vec[i])
	}

	var magnitude float32
	for _, val := range v {
		magnitude += val * val
	}
	magnitude = math32.Sqrt(magnitude)

	if magnitude == 0 {
		return vec, nil
	}

	result := make([]float64, len(vec))
	for i := range v {
		result[i] = float64(v[i] / magnitude)
	}

	return result, nil
}

// DotProduct calculates the dot product of two vectors.
func DotProduct(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, ErrDimensionMismatch
	}
	if len(vec1) == 0 {
		return 0, ErrEmptyVector
	}

	// Convert to float32 for SIMD optimization
	v1 := make([]float32, len(vec1))
	v2 := make([]float32, len(vec2))
	for i := range vec1 {
		v1[i] = float32(vec1[i])
		v2[i] = float32(vec2[i])
	}

	var result float32
	for i := 0; i < len(v1); i++ {
		result += v1[i] * v2[i]
	}

	return float64(result), nil
} 