package apertus

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// validCohereRerankResponse is a minimal valid Cohere rerank JSON response.
var validCohereRerankResponse = map[string]interface{}{
	"id": "rerank-test-id",
	"results": []map[string]interface{}{
		{"index": 0, "relevance_score": 0.95},
		{"index": 1, "relevance_score": 0.30},
	},
	"meta": map[string]interface{}{
		"tokens": map[string]interface{}{
			"input_tokens":  float64(20),
			"output_tokens": float64(0),
		},
	},
}

// validVLLMRerankResponse is a minimal valid vLLM rerank JSON response.
var validVLLMRerankResponse = map[string]interface{}{
	"id":    "rerank-vllm-id",
	"model": "backend-reranker",
	"results": []map[string]interface{}{
		{"index": 0, "relevance_score": 0.85},
		{"index": 1, "relevance_score": 0.42},
	},
	"usage": map[string]interface{}{
		"prompt_tokens": float64(15),
		"total_tokens":  float64(15),
	},
}

func newRerankRequest() *schemas.BifrostRerankRequest {
	return &schemas.BifrostRerankRequest{
		Model: "user-reranker",
		Query: "What is the capital of France?",
		Documents: []schemas.RerankDocument{
			{Text: "Paris is the capital of France."},
			{Text: "Berlin is the capital of Germany."},
		},
	}
}

// TestRerankCohereFormat verifies that rerank works with cohere format (default).
func TestRerankCohereFormat(t *testing.T) {
	t.Parallel()

	var (
		capturedPath      string
		capturedAuthHeader string
		capturedModel     string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedAuthHeader = r.Header.Get("Authorization")

		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]interface{}
		if err := json.Unmarshal(body, &reqBody); err == nil {
			if model, ok := reqBody["model"].(string); ok {
				capturedModel = model
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		responseJSON, _ := json.Marshal(validCohereRerankResponse)
		w.Write(responseJSON)
	}))
	defer server.Close()

	provider := newTestProvider("https://default.should-not-be-used.example.com")

	key := schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		ApertusKeyConfig: &schemas.ApertusKeyConfig{
			Endpoint: server.URL,
			ModelNameMappings: map[string]string{
				"user-reranker": "backend-reranker",
			},
			// RerankFormat not set — defaults to cohere
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := newRerankRequest()

	resp, bifrostErr := provider.Rerank(ctx, key, request)

	require.Nil(t, bifrostErr, "expected no error but got: %v", bifrostErr)
	require.NotNil(t, resp)

	// Verify request was sent to cohere path
	assert.Equal(t, "/v2/rerank", capturedPath,
		"request should be sent to /v2/rerank for cohere format")

	// Verify model name was mapped
	assert.Equal(t, "backend-reranker", capturedModel,
		"request body should use mapped backend model name")

	// Verify auth header
	assert.Equal(t, "Bearer test-api-key", capturedAuthHeader,
		"authorization header should be set with key value")

	// Verify response fields
	assert.Equal(t, "user-reranker", resp.ExtraFields.ModelRequested,
		"ModelRequested should be the original user-facing model name")
	assert.Equal(t, "backend-reranker", resp.ExtraFields.ModelDeployment,
		"ModelDeployment should be the mapped backend model name")
	assert.Equal(t, schemas.RerankRequest, resp.ExtraFields.RequestType)

	// Verify results are present and sorted by relevance score descending
	require.Len(t, resp.Results, 2)
	assert.Equal(t, 0, resp.Results[0].Index)
	assert.Equal(t, 0.95, resp.Results[0].RelevanceScore)
	assert.Equal(t, 1, resp.Results[1].Index)
	assert.Equal(t, 0.30, resp.Results[1].RelevanceScore)

	// Verify usage
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 20, resp.Usage.PromptTokens)
}

// TestRerankVLLMFormat verifies that rerank works with vLLM format.
func TestRerankVLLMFormat(t *testing.T) {
	t.Parallel()

	var capturedPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		responseJSON, _ := json.Marshal(validVLLMRerankResponse)
		w.Write(responseJSON)
	}))
	defer server.Close()

	provider := newTestProvider("https://default.should-not-be-used.example.com")

	key := schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		ApertusKeyConfig: &schemas.ApertusKeyConfig{
			Endpoint:     server.URL,
			RerankFormat: "vllm",
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := newRerankRequest()

	resp, bifrostErr := provider.Rerank(ctx, key, request)

	require.Nil(t, bifrostErr, "expected no error but got: %v", bifrostErr)
	require.NotNil(t, resp)

	// Verify request was sent to vLLM path
	assert.Equal(t, "/v1/rerank", capturedPath,
		"request should be sent to /v1/rerank for vllm format")

	// Verify results
	require.Len(t, resp.Results, 2)
	assert.Equal(t, 0, resp.Results[0].Index)
	assert.Equal(t, 0.85, resp.Results[0].RelevanceScore)

	// Verify usage from vLLM format
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 15, resp.Usage.PromptTokens)
	assert.Equal(t, 15, resp.Usage.TotalTokens)
}

// TestRerankDefaultFormatIsCohere verifies that when no RerankFormat is set,
// the default behavior uses the cohere format.
func TestRerankDefaultFormatIsCohere(t *testing.T) {
	t.Parallel()

	var capturedPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		responseJSON, _ := json.Marshal(validCohereRerankResponse)
		w.Write(responseJSON)
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)

	// Key with no ApertusKeyConfig at all
	key := schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := newRerankRequest()

	resp, bifrostErr := provider.Rerank(ctx, key, request)

	require.Nil(t, bifrostErr, "expected no error but got: %v", bifrostErr)
	require.NotNil(t, resp)

	assert.Equal(t, "/v2/rerank", capturedPath,
		"default format should use /v2/rerank (cohere)")
}

// TestRerankModelNameMapping verifies that model name mapping works correctly
// in both the request sent to the backend and the response metadata.
func TestRerankModelNameMapping(t *testing.T) {
	t.Parallel()

	var capturedModel string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]interface{}
		if err := json.Unmarshal(body, &reqBody); err == nil {
			if model, ok := reqBody["model"].(string); ok {
				capturedModel = model
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		responseJSON, _ := json.Marshal(validCohereRerankResponse)
		w.Write(responseJSON)
	}))
	defer server.Close()

	provider := newTestProvider("https://unused.example.com")

	key := schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		ApertusKeyConfig: &schemas.ApertusKeyConfig{
			Endpoint: server.URL,
			ModelNameMappings: map[string]string{
				"my-reranker": "actual-backend-reranker-v3",
			},
		},
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := &schemas.BifrostRerankRequest{
		Model: "my-reranker",
		Query: "test query",
		Documents: []schemas.RerankDocument{
			{Text: "doc1"},
			{Text: "doc2"},
		},
	}

	resp, bifrostErr := provider.Rerank(ctx, key, request)

	require.Nil(t, bifrostErr)
	require.NotNil(t, resp)

	// Backend should receive the mapped model name
	assert.Equal(t, "actual-backend-reranker-v3", capturedModel)

	// Response should track both original and mapped names
	assert.Equal(t, "my-reranker", resp.ExtraFields.ModelRequested)
	assert.Equal(t, "actual-backend-reranker-v3", resp.ExtraFields.ModelDeployment)

	// Model in the bifrost response should be the original user-facing name
	assert.Equal(t, "my-reranker", resp.Model)
}

// TestRerankOperationNotAllowed verifies that rerank is blocked when not in the allowed requests.
func TestRerankOperationNotAllowed(t *testing.T) {
	t.Parallel()

	provider := NewApertusProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        "https://example.com",
			DefaultRequestTimeoutInSeconds: 10,
		},
		CustomProviderConfig: &schemas.CustomProviderConfig{
			AllowedRequests: &schemas.AllowedRequests{
				ChatCompletion: true,
				// Rerank not allowed
			},
		},
	}, &testLogger{})

	key := schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := newRerankRequest()

	resp, bifrostErr := provider.Rerank(ctx, key, request)

	assert.Nil(t, resp)
	require.NotNil(t, bifrostErr)
}

// TestGetRerankDefaultPath verifies the path selection based on RerankFormat.
func TestGetRerankDefaultPath(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		key      schemas.Key
		expected string
	}{
		{
			name:     "nil ApertusKeyConfig defaults to cohere path",
			key:      schemas.Key{},
			expected: "/v2/rerank",
		},
		{
			name: "empty RerankFormat defaults to cohere path",
			key: schemas.Key{
				ApertusKeyConfig: &schemas.ApertusKeyConfig{},
			},
			expected: "/v2/rerank",
		},
		{
			name: "cohere format uses /v2/rerank",
			key: schemas.Key{
				ApertusKeyConfig: &schemas.ApertusKeyConfig{RerankFormat: "cohere"},
			},
			expected: "/v2/rerank",
		},
		{
			name: "vllm format uses /v1/rerank",
			key: schemas.Key{
				ApertusKeyConfig: &schemas.ApertusKeyConfig{RerankFormat: "vllm"},
			},
			expected: "/v1/rerank",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := getRerankDefaultPath(tt.key)
			assert.Equal(t, tt.expected, result)
		})
	}
}
