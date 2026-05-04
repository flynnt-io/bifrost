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

// testLogger is a no-op logger for use in tests.
type testLogger struct{}

func (l *testLogger) Debug(msg string, args ...any)                     {}
func (l *testLogger) Info(msg string, args ...any)                      {}
func (l *testLogger) Warn(msg string, args ...any)                      {}
func (l *testLogger) Error(msg string, args ...any)                     {}
func (l *testLogger) Fatal(msg string, args ...any)                     {}
func (l *testLogger) SetLevel(level schemas.LogLevel)                   {}
func (l *testLogger) SetOutputType(outputType schemas.LoggerOutputType) {}
func (l *testLogger) LogHTTPRequest(level schemas.LogLevel, msg string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

// newTestProvider creates an ApertusProvider with the given base URL for tests.
func newTestProvider(baseURL string) *ApertusProvider {
	return NewApertusProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        baseURL,
			DefaultRequestTimeoutInSeconds: 10,
		},
	}, &testLogger{})
}

// newKeyWithEndpoint creates a Key with a custom ApertusKeyConfig endpoint.
func newKeyWithEndpoint(endpoint string) schemas.Key {
	return schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		ApertusKeyConfig: &schemas.ApertusKeyConfig{
			Endpoint: endpoint,
		},
	}
}

// newKeyWithMappings creates a Key with ApertusKeyConfig model name mappings.
func newKeyWithMappings(mappings map[string]string) schemas.Key {
	return schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		ApertusKeyConfig: &schemas.ApertusKeyConfig{
			ModelNameMappings: mappings,
		},
	}
}

// newKeyWithEndpointAndMappings creates a Key with both a custom endpoint and model name mappings.
func newKeyWithEndpointAndMappings(endpoint string, mappings map[string]string) schemas.Key {
	return schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		ApertusKeyConfig: &schemas.ApertusKeyConfig{
			Endpoint:          endpoint,
			ModelNameMappings: mappings,
		},
	}
}

// TestGetBaseURL tests the getBaseURL helper with various key configurations.
func TestGetBaseURL(t *testing.T) {
	t.Parallel()

	provider := newTestProvider("https://default.example.com")

	tests := []struct {
		name     string
		key      schemas.Key
		expected string
	}{
		{
			name:     "nil ApertusKeyConfig falls back to provider base URL",
			key:      schemas.Key{},
			expected: "https://default.example.com",
		},
		{
			name: "empty endpoint falls back to provider base URL",
			key: schemas.Key{
				ApertusKeyConfig: &schemas.ApertusKeyConfig{Endpoint: ""},
			},
			expected: "https://default.example.com",
		},
		{
			name:     "custom endpoint is used",
			key:      newKeyWithEndpoint("https://custom.example.com"),
			expected: "https://custom.example.com",
		},
		{
			name:     "trailing slash is trimmed from custom endpoint",
			key:      newKeyWithEndpoint("https://custom.example.com/"),
			expected: "https://custom.example.com",
		},
		{
			name:     "multiple trailing slashes are trimmed",
			key:      newKeyWithEndpoint("https://custom.example.com///"),
			expected: "https://custom.example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := provider.getBaseURL(tt.key)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestGetModelName tests the getModelName helper with various key and model configurations.
func TestGetModelName(t *testing.T) {
	t.Parallel()

	provider := newTestProvider("https://default.example.com")

	tests := []struct {
		name      string
		key       schemas.Key
		userModel string
		wantModel string
	}{
		{
			name:      "nil ApertusKeyConfig returns original model",
			key:       schemas.Key{},
			userModel: "gpt-4o",
			wantModel: "gpt-4o",
		},
		{
			name: "nil ModelNameMappings returns original model",
			key: schemas.Key{
				ApertusKeyConfig: &schemas.ApertusKeyConfig{},
			},
			userModel: "gpt-4o",
			wantModel: "gpt-4o",
		},
		{
			name:      "matching mapping returns backend model",
			key:       newKeyWithMappings(map[string]string{"gpt-4o": "backend-model-v2"}),
			userModel: "gpt-4o",
			wantModel: "backend-model-v2",
		},
		{
			name:      "non-matching mapping returns original model",
			key:       newKeyWithMappings(map[string]string{"gpt-4o": "backend-model-v2"}),
			userModel: "gpt-3.5-turbo",
			wantModel: "gpt-3.5-turbo",
		},
		{
			name:      "model name mapping is case-sensitive",
			key:       newKeyWithMappings(map[string]string{"gpt-4o": "backend-model-v2"}),
			userModel: "GPT-4O",
			wantModel: "GPT-4O",
		},
		{
			name:      "empty mappings returns original model",
			key:       newKeyWithMappings(map[string]string{}),
			userModel: "gpt-4o",
			wantModel: "gpt-4o",
		},
		{
			name:      "multiple mappings selects the correct one",
			key:       newKeyWithMappings(map[string]string{"gpt-4o": "prod-model-a", "claude-3": "prod-model-b"}),
			userModel: "claude-3",
			wantModel: "prod-model-b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := provider.getModelName(tt.key, tt.userModel)
			assert.Equal(t, tt.wantModel, result)
		})
	}
}

// TestListModels tests the ListModels method which is entirely Apertus-specific (no API call).
func TestListModels(t *testing.T) {
	t.Parallel()

	provider := newTestProvider("https://default.example.com")
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	tests := []struct {
		name           string
		keys           []schemas.Key
		wantModelCount int
		wantModels     []string
	}{
		{
			name:           "no keys returns empty model list",
			keys:           []schemas.Key{},
			wantModelCount: 0,
		},
		{
			name: "single key with models",
			keys: []schemas.Key{
				{Models: []string{"gpt-4o", "gpt-3.5-turbo"}},
			},
			wantModelCount: 2,
			wantModels:     []string{"apertus/gpt-4o", "apertus/gpt-3.5-turbo"},
		},
		{
			name: "multiple keys with overlapping models are deduplicated",
			keys: []schemas.Key{
				{Models: []string{"gpt-4o", "gpt-3.5-turbo"}},
				{Models: []string{"gpt-4o", "claude-3"}},
			},
			wantModelCount: 3,
			wantModels:     []string{"apertus/gpt-4o", "apertus/gpt-3.5-turbo", "apertus/claude-3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resp, err := provider.ListModels(ctx, tt.keys, &schemas.BifrostListModelsRequest{})

			require.Nil(t, err)
			require.NotNil(t, resp)
			assert.Equal(t, tt.wantModelCount, len(resp.Data))
			assert.Equal(t, schemas.ListModelsRequest, resp.ExtraFields.RequestType)
			assert.Equal(t, schemas.ModelProvider("apertus"), resp.ExtraFields.Provider)

			// Verify each expected model appears (prefixed with provider name)
			if len(tt.wantModels) > 0 {
				modelIDs := make(map[string]bool)
				for _, m := range resp.Data {
					modelIDs[m.ID] = true
				}
				for _, wantModel := range tt.wantModels {
					assert.True(t, modelIDs[wantModel], "expected model %q in response", wantModel)
				}
			}
		})
	}
}

// validChatCompletionResponse is a minimal valid OpenAI chat completion JSON response
// for use in mock HTTP server tests.
var validChatCompletionResponse = map[string]interface{}{
	"id":      "chatcmpl-test123",
	"object":  "chat.completion",
	"created": 1234567890,
	"model":   "backend-model-v2",
	"choices": []map[string]interface{}{
		{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Hello!",
			},
			"finish_reason": "stop",
		},
	},
	"usage": map[string]interface{}{
		"prompt_tokens":     10,
		"completion_tokens": 5,
		"total_tokens":      15,
	},
}

// TestChatCompletionWithMockServer verifies that Apertus correctly routes
// requests to the key's custom endpoint and applies model name mappings.
func TestChatCompletionWithMockServer(t *testing.T) {
	t.Parallel()

	var (
		capturedPath       string
		capturedAuthHeader string
		capturedModel      string
	)

	// Create a mock HTTP server that captures the request details
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedAuthHeader = r.Header.Get("Authorization")

		// Read and decode the request body to capture model name
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]interface{}
		if err := json.Unmarshal(body, &reqBody); err == nil {
			if model, ok := reqBody["model"].(string); ok {
				capturedModel = model
			}
		}

		// Return a valid chat completion response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		responseJSON, _ := json.Marshal(validChatCompletionResponse)
		w.Write(responseJSON)
	}))
	defer server.Close()

	// Create provider with a different default base URL (to confirm custom endpoint overrides it)
	provider := newTestProvider("https://default.should-not-be-used.example.com")

	// Create key with custom endpoint pointing to mock server and a model name mapping
	key := newKeyWithEndpointAndMappings(server.URL, map[string]string{
		"gpt-4o": "backend-model-v2",
	})

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := &schemas.BifrostChatRequest{
		Model: "gpt-4o",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello!")}},
		},
	}

	resp, bifrostErr := provider.ChatCompletion(ctx, key, request)

	require.Nil(t, bifrostErr, "expected no error but got: %v", bifrostErr)
	require.NotNil(t, resp)

	// Verify request was routed to the custom endpoint (not the default base URL)
	assert.Equal(t, "/v1/chat/completions", capturedPath,
		"request should be sent to /v1/chat/completions")

	// Verify model name was mapped before sending to backend
	assert.Equal(t, "backend-model-v2", capturedModel,
		"request body should use mapped backend model name")

	// Verify auth header is set from key value
	assert.Equal(t, "Bearer test-api-key", capturedAuthHeader,
		"authorization header should be set with key value")

	// Verify response tracks the original user-facing model name
	assert.Equal(t, "gpt-4o", resp.ExtraFields.OriginalModelRequested,
		"OriginalModelRequested should be the original user-facing model name")
}

// TestChatCompletionFallbackToDefaultURL verifies that when a key has no custom
// endpoint, requests use the provider's base URL.
func TestChatCompletionFallbackToDefaultURL(t *testing.T) {
	t.Parallel()

	requestReceived := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestReceived = true
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		responseJSON, _ := json.Marshal(validChatCompletionResponse)
		w.Write(responseJSON)
	}))
	defer server.Close()

	// Provider base URL points to mock server
	provider := newTestProvider(server.URL)

	// Key with NO custom endpoint — should fall back to provider base URL
	key := schemas.Key{
		Value: *schemas.NewEnvVar("test-api-key"),
		// ApertusKeyConfig is nil — no custom endpoint
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := &schemas.BifrostChatRequest{
		Model: "gpt-4o",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello!")}},
		},
	}

	resp, bifrostErr := provider.ChatCompletion(ctx, key, request)

	require.Nil(t, bifrostErr)
	require.NotNil(t, resp)
	assert.True(t, requestReceived, "request should have been received by the server at the provider base URL")
}
