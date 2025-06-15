// Package vectormath provides optimized vector mathematics operations.
package vectormath

import (
	"errors"
	"math"
)

var (
	// ErrDimensionMismatch is returned when vector dimensions don't match
	ErrDimensionMismatch = errors.New("vector dimensions do not match")
	// ErrEmptyVector is returned when a vector is empty
	ErrEmptyVector = errors.New("vector cannot be empty")
)

// CosineSimilarity calculates the cosine similarity between two vectors.
// Optimized to avoid unnecessary conversions and use efficient computation.
func CosineSimilarity(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, ErrDimensionMismatch
	}
	if len(vec1) == 0 {
		return 0, ErrEmptyVector
	}

	var dotProduct, mag1, mag2 float64
	
	// Unroll loop for better performance on small vectors
	i := 0
	for ; i < len(vec1)-3; i += 4 {
		// Process 4 elements at once for better CPU utilization
		dotProduct += vec1[i]*vec2[i] + vec1[i+1]*vec2[i+1] + vec1[i+2]*vec2[i+2] + vec1[i+3]*vec2[i+3]
		mag1 += vec1[i]*vec1[i] + vec1[i+1]*vec1[i+1] + vec1[i+2]*vec1[i+2] + vec1[i+3]*vec1[i+3]
		mag2 += vec2[i]*vec2[i] + vec2[i+1]*vec2[i+1] + vec2[i+2]*vec2[i+2] + vec2[i+3]*vec2[i+3]
	}
	
	// Handle remaining elements
	for ; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		mag1 += vec1[i] * vec1[i]
		mag2 += vec2[i] * vec2[i]
	}

	if mag1 == 0 || mag2 == 0 {
		return 0, nil
	}

	return dotProduct / (math.Sqrt(mag1) * math.Sqrt(mag2)), nil
}

// CosineSimilarityNormalized calculates cosine similarity for pre-normalized vectors.
// Much faster when vectors are already unit length.
func CosineSimilarityNormalized(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, ErrDimensionMismatch
	}
	if len(vec1) == 0 {
		return 0, ErrEmptyVector
	}

	return DotProduct(vec1, vec2)
}

// Normalize normalizes a vector to unit length in-place for efficiency.
func Normalize(vec []float64) error {
	if len(vec) == 0 {
		return ErrEmptyVector
	}

	var magnitude float64
	for _, val := range vec {
		magnitude += val * val
	}

	if magnitude == 0 {
		return nil
	}

	magnitude = math.Sqrt(magnitude)
	for i := range vec {
		vec[i] /= magnitude
	}

	return nil
}

// NormalizeCopy returns a normalized copy of the vector.
func NormalizeCopy(vec []float64) ([]float64, error) {
	result := make([]float64, len(vec))
	copy(result, vec)
	err := Normalize(result)
	return result, err
}

// DotProduct calculates the dot product of two vectors efficiently.
func DotProduct(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, ErrDimensionMismatch
	}
	if len(vec1) == 0 {
		return 0, ErrEmptyVector
	}

	var result float64
	i := 0
	
	// Unroll loop for better performance
	for ; i < len(vec1)-3; i += 4 {
		result += vec1[i]*vec2[i] + vec1[i+1]*vec2[i+1] + vec1[i+2]*vec2[i+2] + vec1[i+3]*vec2[i+3]
	}
	
	// Handle remaining elements
	for ; i < len(vec1); i++ {
		result += vec1[i] * vec2[i]
	}

	return result, nil
}

// Magnitude calculates the magnitude (L2 norm) of a vector.
func Magnitude(vec []float64) (float64, error) {
	if len(vec) == 0 {
		return 0, ErrEmptyVector
	}

	var sum float64
	for _, val := range vec {
		sum += val * val
	}
	
	return math.Sqrt(sum), nil
}

// BatchCosineSimilarity calculates cosine similarity between one query vector
// and multiple candidate vectors efficiently.
func BatchCosineSimilarity(query []float64, candidates [][]float64) ([]float64, error) {
	if len(query) == 0 {
		return nil, ErrEmptyVector
	}

	results := make([]float64, len(candidates))
	
	// Pre-calculate query magnitude
	var queryMag float64
	for _, val := range query {
		queryMag += val * val
	}
	
	if queryMag == 0 {
		return results, nil
	}
	
	queryMag = math.Sqrt(queryMag)
	
	for i, candidate := range candidates {
		if len(candidate) != len(query) {
			return nil, ErrDimensionMismatch
		}
		
		var dotProduct, candidateMag float64
		for j := range query {
			dotProduct += query[j] * candidate[j]
			candidateMag += candidate[j] * candidate[j]
		}
		
		if candidateMag == 0 {
			results[i] = 0
		} else {
			results[i] = dotProduct / (queryMag * math.Sqrt(candidateMag))
		}
	}
	
	return results, nil
} 