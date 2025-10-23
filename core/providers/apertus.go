// Package providers implements various LLM providers and their utility functions.
// This file contains the Apertus provider implementation.
package providers

import (
	"context"
	"net/http"
	"strings"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// ApertusProvider implements the Provider interface for Apertus API.
// It is similar to OpenAI but allows per-key custom endpoint URLs.
type ApertusProvider struct {
	logger               schemas.Logger                // Logger for provider operations
	client               *fasthttp.Client              // HTTP client for API requests
	streamClient         *http.Client                  // HTTP client for streaming requests
	networkConfig        schemas.NetworkConfig         // Network configuration including extra headers
	sendBackRawResponse  bool                          // Whether to include raw response in BifrostResponse
	customProviderConfig *schemas.CustomProviderConfig // Custom provider config
}

// NewApertusProvider creates a new Apertus provider instance.
// It initializes the HTTP client with the provided configuration and sets up response pools.
// The client is configured with timeouts, concurrency limits, and optional proxy settings.
func NewApertusProvider(config *schemas.ProviderConfig, logger schemas.Logger) *ApertusProvider {
	config.CheckAndSetDefaults()

	client := &fasthttp.Client{
		ReadTimeout:     time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds),
		WriteTimeout:    time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds),
		MaxConnsPerHost: config.ConcurrencyAndBufferSize.Concurrency,
	}

	// Initialize streaming HTTP client
	streamClient := &http.Client{
		Timeout: time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds),
	}

	// Configure proxy if provided
	client = configureProxy(client, config.ProxyConfig, logger)

	// Set default BaseURL if not provided (falls back to OpenAI)
	if config.NetworkConfig.BaseURL == "" {
		config.NetworkConfig.BaseURL = "https://api.openai.com"
	}
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	return &ApertusProvider{
		logger:               logger,
		client:               client,
		streamClient:         streamClient,
		networkConfig:        config.NetworkConfig,
		sendBackRawResponse:  config.SendBackRawResponse,
		customProviderConfig: config.CustomProviderConfig,
	}
}

// GetProviderKey returns the provider identifier for Apertus.
func (provider *ApertusProvider) GetProviderKey() schemas.ModelProvider {
	return getProviderName(schemas.Apertus, provider.customProviderConfig)
}

// getBaseURL returns the effective base URL for the given key.
// If the key has a custom endpoint configured, it uses that; otherwise falls back to the provider's base URL.
func (provider *ApertusProvider) getBaseURL(key schemas.Key) string {
	if key.ApertusKeyConfig != nil && key.ApertusKeyConfig.Endpoint != "" {
		return strings.TrimRight(key.ApertusKeyConfig.Endpoint, "/")
	}
	return provider.networkConfig.BaseURL
}

// TextCompletion performs a text completion request to Apertus API.
func (provider *ApertusProvider) TextCompletion(ctx context.Context, key schemas.Key, request *schemas.BifrostTextCompletionRequest) (*schemas.BifrostTextCompletionResponse, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TextCompletionRequest); err != nil {
		return nil, err
	}
	return handleOpenAITextCompletionRequest(
		ctx,
		provider.client,
		provider.getBaseURL(key)+"/v1/completions",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		provider.sendBackRawResponse,
		provider.logger,
	)
}

// TextCompletionStream performs a streaming text completion request to Apertus API.
func (provider *ApertusProvider) TextCompletionStream(ctx context.Context, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostTextCompletionRequest) (chan *schemas.BifrostStream, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TextCompletionStreamRequest); err != nil {
		return nil, err
	}
	return handleOpenAITextCompletionStreaming(
		ctx,
		provider.streamClient,
		provider.getBaseURL(key)+"/v1/completions",
		request,
		map[string]string{"Authorization": "Bearer " + key.Value},
		provider.networkConfig.ExtraHeaders,
		provider.sendBackRawResponse,
		provider.GetProviderKey(),
		postHookRunner,
		provider.logger,
	)
}

// ChatCompletion performs a chat completion request to the Apertus API.
func (provider *ApertusProvider) ChatCompletion(ctx context.Context, key schemas.Key, request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ChatCompletionRequest); err != nil {
		return nil, err
	}

	return handleOpenAIChatCompletionRequest(
		ctx,
		provider.client,
		provider.getBaseURL(key)+"/v1/chat/completions",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.sendBackRawResponse,
		provider.GetProviderKey(),
		provider.logger,
	)
}

// ChatCompletionStream handles streaming for Apertus chat completions.
func (provider *ApertusProvider) ChatCompletionStream(ctx context.Context, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostChatRequest) (chan *schemas.BifrostStream, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ChatCompletionStreamRequest); err != nil {
		return nil, err
	}

	return handleOpenAIChatCompletionStreaming(
		ctx,
		provider.streamClient,
		provider.getBaseURL(key)+"/v1/chat/completions",
		request,
		map[string]string{"Authorization": "Bearer " + key.Value},
		provider.networkConfig.ExtraHeaders,
		provider.sendBackRawResponse,
		provider.GetProviderKey(),
		postHookRunner,
		provider.logger,
	)
}

// Responses performs a responses request to the Apertus API.
func (provider *ApertusProvider) Responses(ctx context.Context, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ResponsesRequest); err != nil {
		return nil, err
	}

	return handleOpenAIResponsesRequest(
		ctx,
		provider.client,
		provider.getBaseURL(key)+"/v1/responses",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.sendBackRawResponse,
		provider.GetProviderKey(),
		provider.logger,
	)
}

// ResponsesStream performs a streaming responses request to the Apertus API.
func (provider *ApertusProvider) ResponsesStream(ctx context.Context, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStream, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ResponsesStreamRequest); err != nil {
		return nil, err
	}

	return handleOpenAIResponsesStreaming(
		ctx,
		provider.streamClient,
		provider.getBaseURL(key)+"/v1/responses",
		request,
		map[string]string{"Authorization": "Bearer " + key.Value},
		provider.networkConfig.ExtraHeaders,
		provider.sendBackRawResponse,
		provider.GetProviderKey(),
		postHookRunner,
		provider.logger,
	)
}

// Embedding generates embeddings for the given input text(s).
func (provider *ApertusProvider) Embedding(ctx context.Context, key schemas.Key, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.EmbeddingRequest); err != nil {
		return nil, err
	}

	return handleOpenAIEmbeddingRequest(
		ctx,
		provider.client,
		provider.getBaseURL(key)+"/v1/embeddings",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		provider.sendBackRawResponse,
		provider.logger,
	)
}

// Speech handles non-streaming speech synthesis requests.
func (provider *ApertusProvider) Speech(ctx context.Context, key schemas.Key, request *schemas.BifrostSpeechRequest) (*schemas.BifrostSpeechResponse, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.SpeechRequest); err != nil {
		return nil, err
	}

	// Delegate to the OpenAI provider's speech implementation by reusing its logic
	// Create a temporary OpenAI provider with the custom endpoint
	tempProvider := &OpenAIProvider{
		logger:               provider.logger,
		client:               provider.client,
		streamClient:         provider.streamClient,
		networkConfig:        schemas.NetworkConfig{
			BaseURL:      provider.getBaseURL(key),
			ExtraHeaders: provider.networkConfig.ExtraHeaders,
			DefaultRequestTimeoutInSeconds: provider.networkConfig.DefaultRequestTimeoutInSeconds,
			MaxRetries:                    provider.networkConfig.MaxRetries,
			RetryBackoffInitial:           provider.networkConfig.RetryBackoffInitial,
			RetryBackoffMax:               provider.networkConfig.RetryBackoffMax,
		},
		sendBackRawResponse:  provider.sendBackRawResponse,
		customProviderConfig: nil, // Don't pass custom config to avoid confusion
	}

	// Call OpenAI's Speech method but return response with Apertus provider name
	response, err := tempProvider.Speech(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// SpeechStream handles streaming for speech synthesis.
func (provider *ApertusProvider) SpeechStream(ctx context.Context, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostSpeechRequest) (chan *schemas.BifrostStream, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.SpeechStreamRequest); err != nil {
		return nil, err
	}

	// Delegate to the OpenAI provider's speech streaming implementation
	tempProvider := &OpenAIProvider{
		logger:               provider.logger,
		client:               provider.client,
		streamClient:         provider.streamClient,
		networkConfig:        schemas.NetworkConfig{
			BaseURL:      provider.getBaseURL(key),
			ExtraHeaders: provider.networkConfig.ExtraHeaders,
			DefaultRequestTimeoutInSeconds: provider.networkConfig.DefaultRequestTimeoutInSeconds,
			MaxRetries:                    provider.networkConfig.MaxRetries,
			RetryBackoffInitial:           provider.networkConfig.RetryBackoffInitial,
			RetryBackoffMax:               provider.networkConfig.RetryBackoffMax,
		},
		sendBackRawResponse:  provider.sendBackRawResponse,
		customProviderConfig: nil,
	}

	return tempProvider.SpeechStream(ctx, postHookRunner, key, request)
}

// Transcription handles non-streaming transcription requests.
func (provider *ApertusProvider) Transcription(ctx context.Context, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TranscriptionRequest); err != nil {
		return nil, err
	}

	// Delegate to the OpenAI provider's transcription implementation
	tempProvider := &OpenAIProvider{
		logger:               provider.logger,
		client:               provider.client,
		streamClient:         provider.streamClient,
		networkConfig:        schemas.NetworkConfig{
			BaseURL:      provider.getBaseURL(key),
			ExtraHeaders: provider.networkConfig.ExtraHeaders,
			DefaultRequestTimeoutInSeconds: provider.networkConfig.DefaultRequestTimeoutInSeconds,
			MaxRetries:                    provider.networkConfig.MaxRetries,
			RetryBackoffInitial:           provider.networkConfig.RetryBackoffInitial,
			RetryBackoffMax:               provider.networkConfig.RetryBackoffMax,
		},
		sendBackRawResponse:  provider.sendBackRawResponse,
		customProviderConfig: nil,
	}

	response, err := tempProvider.Transcription(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// TranscriptionStream performs a streaming transcription request to the Apertus API.
func (provider *ApertusProvider) TranscriptionStream(ctx context.Context, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (chan *schemas.BifrostStream, *schemas.BifrostError) {
	if err := checkOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TranscriptionStreamRequest); err != nil {
		return nil, err
	}

	// Delegate to the OpenAI provider's transcription streaming implementation
	tempProvider := &OpenAIProvider{
		logger:               provider.logger,
		client:               provider.client,
		streamClient:         provider.streamClient,
		networkConfig:        schemas.NetworkConfig{
			BaseURL:      provider.getBaseURL(key),
			ExtraHeaders: provider.networkConfig.ExtraHeaders,
			DefaultRequestTimeoutInSeconds: provider.networkConfig.DefaultRequestTimeoutInSeconds,
			MaxRetries:                    provider.networkConfig.MaxRetries,
			RetryBackoffInitial:           provider.networkConfig.RetryBackoffInitial,
			RetryBackoffMax:               provider.networkConfig.RetryBackoffMax,
		},
		sendBackRawResponse:  provider.sendBackRawResponse,
		customProviderConfig: nil,
	}

	return tempProvider.TranscriptionStream(ctx, postHookRunner, key, request)
}
