// Package apertus implements the Apertus provider for the Bifrost framework.
// This file contains the Apertus provider implementation.
package apertus

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/maximhq/bifrost/core/providers/cohere"
	"github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/providers/vllm"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// ApertusProvider implements the Provider interface for Apertus API.
// It is similar to OpenAI but allows per-key custom endpoint URLs.
type ApertusProvider struct {
	logger               schemas.Logger                // Logger for provider operations
	client               *fasthttp.Client              // HTTP client for API requests
	networkConfig        schemas.NetworkConfig         // Network configuration including extra headers
	sendBackRawRequest   bool                          // Whether to include raw request in BifrostResponse
	sendBackRawResponse  bool                          // Whether to include raw response in BifrostResponse
	customProviderConfig *schemas.CustomProviderConfig // Custom provider config
}

// NewApertusProvider creates a new Apertus provider instance.
// It initializes the HTTP client with the provided configuration and sets up response pools.
// The client is configured with timeouts, concurrency limits, and optional proxy settings.
func NewApertusProvider(config *schemas.ProviderConfig, logger schemas.Logger) *ApertusProvider {
	config.CheckAndSetDefaults()

	client := &fasthttp.Client{
		ReadTimeout:         time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds),
		WriteTimeout:        time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds),
		MaxConnsPerHost:     5000,
		MaxIdleConnDuration: 60 * time.Second,
		MaxConnWaitTimeout:  10 * time.Second,
	}

	// Configure proxy if provided
	client = providerUtils.ConfigureProxy(client, config.ProxyConfig, logger)

	// Set default BaseURL if not provided (falls back to OpenAI)
	if config.NetworkConfig.BaseURL == "" {
		config.NetworkConfig.BaseURL = "https://api.openai.com"
	}
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	return &ApertusProvider{
		logger:               logger,
		client:               client,
		networkConfig:        config.NetworkConfig,
		sendBackRawRequest:   config.SendBackRawRequest,
		sendBackRawResponse:  config.SendBackRawResponse,
		customProviderConfig: config.CustomProviderConfig,
	}
}

// GetProviderKey returns the provider identifier for Apertus.
func (provider *ApertusProvider) GetProviderKey() schemas.ModelProvider {
	return providerUtils.GetProviderName(schemas.Apertus, provider.customProviderConfig)
}

// getBaseURL returns the effective base URL for the given key.
// If the key has a custom endpoint configured, it uses that; otherwise falls back to the provider's base URL.
func (provider *ApertusProvider) getBaseURL(key schemas.Key) string {
	if key.ApertusKeyConfig != nil && key.ApertusKeyConfig.Endpoint != "" {
		return strings.TrimRight(key.ApertusKeyConfig.Endpoint, "/")
	}
	return provider.networkConfig.BaseURL
}

// buildRequestURL constructs the full request URL using the provider's configuration.
// It uses the key's custom endpoint if configured, then applies any custom request path overrides.
func (provider *ApertusProvider) buildRequestURL(ctx *schemas.BifrostContext, key schemas.Key, defaultPath string, requestType schemas.RequestType) string {
	baseURL := provider.getBaseURL(key)
	path, isCompleteURL := providerUtils.GetRequestPath(ctx, defaultPath, provider.customProviderConfig, requestType)
	if isCompleteURL {
		return path
	}
	return baseURL + path
}

// getModelName returns the mapped model name if a mapping exists, otherwise returns the original model name.
// This allows transparent model name mapping without requiring configuration (fallback to original).
func (provider *ApertusProvider) getModelName(key schemas.Key, userModel string) string {
	if key.ApertusKeyConfig != nil && key.ApertusKeyConfig.ModelNameMappings != nil {
		if backendModel, ok := key.ApertusKeyConfig.ModelNameMappings[userModel]; ok {
			provider.logger.Debug(fmt.Sprintf("Apertus: Mapped model '%s' to '%s'", userModel, backendModel))
			return backendModel
		}
	}
	return userModel
}

// createDelegateForKey creates a temporary OpenAI provider configured with the given key's endpoint.
// This is used to delegate operations that are not directly implemented in Apertus.
func (provider *ApertusProvider) createDelegateForKey(key schemas.Key) *openai.OpenAIProvider {
	config := &schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        provider.getBaseURL(key),
			ExtraHeaders:                   provider.networkConfig.ExtraHeaders,
			DefaultRequestTimeoutInSeconds: provider.networkConfig.DefaultRequestTimeoutInSeconds,
			MaxRetries:                     provider.networkConfig.MaxRetries,
			RetryBackoffInitial:            provider.networkConfig.RetryBackoffInitial,
			RetryBackoffMax:                provider.networkConfig.RetryBackoffMax,
		},
		SendBackRawRequest:  provider.sendBackRawRequest,
		SendBackRawResponse: provider.sendBackRawResponse,
	}
	return openai.NewOpenAIProvider(config, provider.logger)
}

// ListModels returns a static list of models configured for the keys.
// Unlike other providers, Apertus does not call the /v1/models API endpoint.
// Instead, it returns the models configured in the key configuration.
func (provider *ApertusProvider) ListModels(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ListModelsRequest); err != nil {
		return nil, err
	}

	providerName := provider.GetProviderKey()

	// Collect all unique models from all keys
	modelSet := make(map[string]bool)
	for _, key := range keys {
		for _, model := range key.Models {
			modelSet[model] = true
		}
	}

	// Convert to slice and sort for consistent output
	models := make([]string, 0, len(modelSet))
	for model := range modelSet {
		models = append(models, model)
	}

	// Convert to Model format with provider prefix
	modelInfos := make([]schemas.Model, len(models))
	ownedBy := "system"
	for i, model := range models {
		modelInfos[i] = schemas.Model{
			ID:      string(providerName) + "/" + model,
			OwnedBy: &ownedBy,
		}
	}

	response := &schemas.BifrostListModelsResponse{
		Data: modelInfos,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Provider:    providerName,
			RequestType: schemas.ListModelsRequest,
			Latency:     0, // No actual API call made
		},
	}

	return response, nil
}

// TextCompletion performs a text completion request to Apertus API.
func (provider *ApertusProvider) TextCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTextCompletionRequest) (*schemas.BifrostTextCompletionResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TextCompletionRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	response, err := openai.HandleOpenAITextCompletionRequest(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/completions", schemas.TextCompletionRequest),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		nil,
		nil,
		provider.logger,
	)

	if response != nil {
		response.ExtraFields.OriginalModelRequested = originalModel
	}

	return response, err
}

// TextCompletionStream performs a streaming text completion request to Apertus API.
func (provider *ApertusProvider) TextCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostTextCompletionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TextCompletionStreamRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	postResponseConverter := func(response *schemas.BifrostTextCompletionResponse) *schemas.BifrostTextCompletionResponse {
		response.ExtraFields.OriginalModelRequested = originalModel
		return response
	}

	var authHeader map[string]string
	if key.Value.GetValue() != "" {
		authHeader = map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
	}

	return openai.HandleOpenAITextCompletionStreaming(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/completions", schemas.TextCompletionStreamRequest),
		request,
		authHeader,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		nil,
		postHookRunner,
		nil,
		postResponseConverter,
		provider.logger,
		postHookSpanFinalizer,
	)
}

// ChatCompletion performs a chat completion request to the Apertus API.
func (provider *ApertusProvider) ChatCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ChatCompletionRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	response, err := openai.HandleOpenAIChatCompletionRequest(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/chat/completions", schemas.ChatCompletionRequest),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		nil,
		nil,
		provider.logger,
	)

	// Set OriginalModelRequested for metrics
	if response != nil {
		response.ExtraFields.OriginalModelRequested = originalModel
	}

	return response, err
}

// ChatCompletionStream handles streaming for Apertus chat completions.
func (provider *ApertusProvider) ChatCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ChatCompletionStreamRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	postResponseConverter := func(response *schemas.BifrostChatResponse) *schemas.BifrostChatResponse {
		response.ExtraFields.OriginalModelRequested = originalModel
		return response
	}

	var authHeader map[string]string
	if key.Value.GetValue() != "" {
		authHeader = map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
	}

	return openai.HandleOpenAIChatCompletionStreaming(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/chat/completions", schemas.ChatCompletionStreamRequest),
		request,
		authHeader,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		postHookRunner,
		nil, // customRequestConverter
		nil, // customResponseHandler
		nil, // customErrorConverter
		nil, // postRequestConverter
		postResponseConverter,
		provider.logger,
		postHookSpanFinalizer,
	)
}

// Responses performs a responses request to the Apertus API.
func (provider *ApertusProvider) Responses(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ResponsesRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	response, err := openai.HandleOpenAIResponsesRequest(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/responses", schemas.ResponsesRequest),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		nil,
		nil,
		provider.logger,
	)

	// Set OriginalModelRequested for metrics
	if response != nil {
		response.ExtraFields.OriginalModelRequested = originalModel
	}

	return response, err
}

// ResponsesStream performs a streaming responses request to the Apertus API.
func (provider *ApertusProvider) ResponsesStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ResponsesStreamRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	postResponseConverter := func(response *schemas.BifrostResponsesStreamResponse) *schemas.BifrostResponsesStreamResponse {
		response.ExtraFields.OriginalModelRequested = originalModel
		return response
	}

	var authHeader map[string]string
	if key.Value.GetValue() != "" {
		authHeader = map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
	}

	return openai.HandleOpenAIResponsesStreaming(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/responses", schemas.ResponsesStreamRequest),
		request,
		authHeader,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		postHookRunner,
		nil, // customResponseHandler
		nil, // customErrorConverter
		nil, // postRequestConverter
		postResponseConverter,
		provider.logger,
		postHookSpanFinalizer,
	)
}

// CountTokens performs a count tokens request to the Apertus API.
func (provider *ApertusProvider) CountTokens(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostCountTokensResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.CountTokensRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.CountTokens(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// Embedding generates embeddings for the given input text(s).
func (provider *ApertusProvider) Embedding(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.EmbeddingRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	response, err := openai.HandleOpenAIEmbeddingRequest(
		ctx,
		provider.client,
		provider.buildRequestURL(ctx, key, "/v1/embeddings", schemas.EmbeddingRequest),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		nil,
		provider.logger,
	)

	// Set OriginalModelRequested for metrics
	if response != nil {
		response.ExtraFields.OriginalModelRequested = originalModel
	}

	return response, err
}

// getRerankDefaultPath returns the default rerank endpoint path based on the key's rerank format.
// Returns "/v2/rerank" for cohere format (default) or "/v1/rerank" for vllm format.
func getRerankDefaultPath(key schemas.Key) string {
	if key.ApertusKeyConfig != nil && key.ApertusKeyConfig.RerankFormat == "vllm" {
		return "/v1/rerank"
	}
	return "/v2/rerank"
}

// Rerank performs a rerank request to the Apertus API.
// It supports both Cohere (/v2/rerank) and vLLM (/v1/rerank) wire formats,
// selected via the key's RerankFormat configuration.
func (provider *ApertusProvider) Rerank(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostRerankRequest) (*schemas.BifrostRerankResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.RerankRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	// Determine format and select converter
	isVLLM := key.ApertusKeyConfig != nil && key.ApertusKeyConfig.RerankFormat == "vllm"

	var converter func() (providerUtils.RequestBodyWithExtraParams, error)
	if isVLLM {
		converter = func() (providerUtils.RequestBodyWithExtraParams, error) {
			return vllm.ToVLLMRerankRequest(request), nil
		}
	} else {
		converter = func() (providerUtils.RequestBodyWithExtraParams, error) {
			return cohere.ToCohereRerankRequest(request), nil
		}
	}

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		converter,
	)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	// Build URL and make HTTP request
	url := provider.buildRequestURL(ctx, key, getRerankDefaultPath(key), schemas.RerankRequest)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	req.SetRequestURI(url)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	if key.Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}
	req.SetBody(jsonData)

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse)
	}

	// Handle error responses
	if resp.StatusCode() != fasthttp.StatusOK {
		apiErr := openai.ParseOpenAIError(resp)
		return nil, providerUtils.EnrichError(ctx, apiErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse)
	}

	// Decode response body
	body, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err)
	}
	bodyCopy := append([]byte(nil), body...)

	sendBackRawReq := providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest)
	sendBackRawResp := providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse)

	returnDocuments := request.Params != nil && request.Params.ReturnDocuments != nil && *request.Params.ReturnDocuments

	// Parse response using format-specific converter
	var bifrostResponse *schemas.BifrostRerankResponse
	var rawRequest, rawResponse interface{}

	if isVLLM {
		responsePayload := make(map[string]interface{})
		rawRequest, rawResponse, bifrostErr = providerUtils.HandleProviderResponse(bodyCopy, &responsePayload, jsonData, sendBackRawReq, sendBackRawResp)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, bodyCopy, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}

		var convErr error
		bifrostResponse, convErr = vllm.ToBifrostRerankResponse(responsePayload, request.Documents, returnDocuments)
		if convErr != nil {
			return nil, providerUtils.EnrichError(ctx,
				providerUtils.NewBifrostOperationError("error converting rerank response", convErr),
				jsonData, bodyCopy, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}
	} else {
		cohereResponse := &cohere.CohereRerankResponse{}
		rawRequest, rawResponse, bifrostErr = providerUtils.HandleProviderResponse(bodyCopy, cohereResponse, jsonData, sendBackRawReq, sendBackRawResp)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, bodyCopy, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}

		bifrostResponse = cohereResponse.ToBifrostRerankResponse(request.Documents, returnDocuments)
	}

	// Set response fields
	bifrostResponse.Model = originalModel
	bifrostResponse.ExtraFields.Provider = provider.GetProviderKey()
	bifrostResponse.ExtraFields.OriginalModelRequested = originalModel
	bifrostResponse.ExtraFields.RequestType = schemas.RerankRequest
	bifrostResponse.ExtraFields.Latency = latency.Milliseconds()

	if sendBackRawReq {
		bifrostResponse.ExtraFields.RawRequest = rawRequest
	}
	if sendBackRawResp {
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

// Speech handles non-streaming speech synthesis requests.
func (provider *ApertusProvider) Speech(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostSpeechRequest) (*schemas.BifrostSpeechResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.SpeechRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	delegate := provider.createDelegateForKey(key)
	response, err := delegate.Speech(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
		response.ExtraFields.OriginalModelRequested = originalModel
	}
	return response, nil
}

// SpeechStream handles streaming for speech synthesis.
func (provider *ApertusProvider) SpeechStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostSpeechRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.SpeechStreamRequest); err != nil {
		return nil, err
	}

	// Apply model name mapping
	request.Model = provider.getModelName(key, request.Model)

	delegate := provider.createDelegateForKey(key)
	return delegate.SpeechStream(ctx, postHookRunner, postHookSpanFinalizer, key, request)
}

// Transcription handles non-streaming transcription requests.
func (provider *ApertusProvider) Transcription(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TranscriptionRequest); err != nil {
		return nil, err
	}

	// Store original model name before mapping
	originalModel := request.Model

	// Apply model name mapping
	mappedModel := provider.getModelName(key, request.Model)
	request.Model = mappedModel

	delegate := provider.createDelegateForKey(key)
	response, err := delegate.Transcription(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
		response.ExtraFields.OriginalModelRequested = originalModel
	}
	return response, nil
}

// TranscriptionStream performs a streaming transcription request to the Apertus API.
func (provider *ApertusProvider) TranscriptionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostTranscriptionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.TranscriptionStreamRequest); err != nil {
		return nil, err
	}

	// Apply model name mapping
	request.Model = provider.getModelName(key, request.Model)

	delegate := provider.createDelegateForKey(key)
	return delegate.TranscriptionStream(ctx, postHookRunner, postHookSpanFinalizer, key, request)
}

// ImageGeneration performs an image generation request to the Apertus API.
func (provider *ApertusProvider) ImageGeneration(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageGenerationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ImageGenerationRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.ImageGeneration(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ImageGenerationStream performs a streaming image generation request to the Apertus API.
func (provider *ApertusProvider) ImageGenerationStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostImageGenerationRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ImageGenerationStreamRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	return delegate.ImageGenerationStream(ctx, postHookRunner, postHookSpanFinalizer, key, request)
}

// ImageEdit performs an image edit request to the Apertus API.
func (provider *ApertusProvider) ImageEdit(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageEditRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ImageEditRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.ImageEdit(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ImageEditStream performs a streaming image edit request to the Apertus API.
func (provider *ApertusProvider) ImageEditStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostImageEditRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ImageEditStreamRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	return delegate.ImageEditStream(ctx, postHookRunner, postHookSpanFinalizer, key, request)
}

// ImageVariation performs an image variation request to the Apertus API.
func (provider *ApertusProvider) ImageVariation(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageVariationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ImageVariationRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.ImageVariation(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// VideoGeneration performs a video generation request to the Apertus API.
func (provider *ApertusProvider) VideoGeneration(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostVideoGenerationRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.VideoGenerationRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.VideoGeneration(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// VideoRetrieve retrieves a video from the Apertus API.
func (provider *ApertusProvider) VideoRetrieve(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostVideoRetrieveRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.VideoRetrieveRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.VideoRetrieve(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// VideoDownload downloads a video from the Apertus API.
func (provider *ApertusProvider) VideoDownload(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostVideoDownloadRequest) (*schemas.BifrostVideoDownloadResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.VideoDownloadRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.VideoDownload(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// VideoDelete deletes a video from the Apertus API.
func (provider *ApertusProvider) VideoDelete(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostVideoDeleteRequest) (*schemas.BifrostVideoDeleteResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.VideoDeleteRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.VideoDelete(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// VideoList lists videos from the Apertus API.
func (provider *ApertusProvider) VideoList(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostVideoListRequest) (*schemas.BifrostVideoListResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.VideoListRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.VideoList(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// VideoRemix remixes a video from the Apertus API.
func (provider *ApertusProvider) VideoRemix(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostVideoRemixRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.VideoRemixRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.VideoRemix(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// BatchCreate creates a new batch job for asynchronous processing.
func (provider *ApertusProvider) BatchCreate(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostBatchCreateRequest) (*schemas.BifrostBatchCreateResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.BatchCreateRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.BatchCreate(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// BatchList lists batch jobs using the first key's endpoint.
func (provider *ApertusProvider) BatchList(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchListRequest) (*schemas.BifrostBatchListResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.BatchListRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.BatchList(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// BatchRetrieve retrieves a specific batch job using the first key's endpoint.
func (provider *ApertusProvider) BatchRetrieve(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchRetrieveRequest) (*schemas.BifrostBatchRetrieveResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.BatchRetrieveRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.BatchRetrieve(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// BatchCancel cancels a batch job using the first key's endpoint.
func (provider *ApertusProvider) BatchCancel(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchCancelRequest) (*schemas.BifrostBatchCancelResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.BatchCancelRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.BatchCancel(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// BatchResults retrieves results from a completed batch job using the first key's endpoint.
func (provider *ApertusProvider) BatchResults(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchResultsRequest) (*schemas.BifrostBatchResultsResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.BatchResultsRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.BatchResults(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// FileUpload uploads a file to the Apertus API.
func (provider *ApertusProvider) FileUpload(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostFileUploadRequest) (*schemas.BifrostFileUploadResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.FileUploadRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.FileUpload(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// FileList lists files using the first key's endpoint.
func (provider *ApertusProvider) FileList(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostFileListRequest) (*schemas.BifrostFileListResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.FileListRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.FileList(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// FileRetrieve retrieves file metadata using the first key's endpoint.
func (provider *ApertusProvider) FileRetrieve(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostFileRetrieveRequest) (*schemas.BifrostFileRetrieveResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.FileRetrieveRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.FileRetrieve(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// FileDelete deletes a file using the first key's endpoint.
func (provider *ApertusProvider) FileDelete(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostFileDeleteRequest) (*schemas.BifrostFileDeleteResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.FileDeleteRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.FileDelete(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// FileContent downloads file content using the first key's endpoint.
func (provider *ApertusProvider) FileContent(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostFileContentRequest) (*schemas.BifrostFileContentResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.FileContentRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.FileContent(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerCreate creates a new container via the Apertus API.
func (provider *ApertusProvider) ContainerCreate(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostContainerCreateRequest) (*schemas.BifrostContainerCreateResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerCreateRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.ContainerCreate(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerList lists containers using the first key's endpoint.
func (provider *ApertusProvider) ContainerList(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerListRequest) (*schemas.BifrostContainerListResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerListRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerList(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerRetrieve retrieves a specific container using the first key's endpoint.
func (provider *ApertusProvider) ContainerRetrieve(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerRetrieveRequest) (*schemas.BifrostContainerRetrieveResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerRetrieveRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerRetrieve(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerDelete deletes a container using the first key's endpoint.
func (provider *ApertusProvider) ContainerDelete(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerDeleteRequest) (*schemas.BifrostContainerDeleteResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerDeleteRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerDelete(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerFileCreate creates a file in a container via the Apertus API.
func (provider *ApertusProvider) ContainerFileCreate(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostContainerFileCreateRequest) (*schemas.BifrostContainerFileCreateResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerFileCreateRequest); err != nil {
		return nil, err
	}
	delegate := provider.createDelegateForKey(key)
	response, err := delegate.ContainerFileCreate(ctx, key, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerFileList lists files in a container using the first key's endpoint.
func (provider *ApertusProvider) ContainerFileList(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerFileListRequest) (*schemas.BifrostContainerFileListResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerFileListRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerFileList(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerFileRetrieve retrieves a file from a container using the first key's endpoint.
func (provider *ApertusProvider) ContainerFileRetrieve(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerFileRetrieveRequest) (*schemas.BifrostContainerFileRetrieveResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerFileRetrieveRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerFileRetrieve(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerFileContent retrieves the content of a file from a container using the first key's endpoint.
func (provider *ApertusProvider) ContainerFileContent(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerFileContentRequest) (*schemas.BifrostContainerFileContentResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerFileContentRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerFileContent(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// ContainerFileDelete deletes a file from a container using the first key's endpoint.
func (provider *ApertusProvider) ContainerFileDelete(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostContainerFileDeleteRequest) (*schemas.BifrostContainerFileDeleteResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Apertus, provider.customProviderConfig, schemas.ContainerFileDeleteRequest); err != nil {
		return nil, err
	}
	var firstKey schemas.Key
	if len(keys) > 0 {
		firstKey = keys[0]
	}
	delegate := provider.createDelegateForKey(firstKey)
	response, err := delegate.ContainerFileDelete(ctx, keys, request)
	if err != nil {
		err.ExtraFields.Provider = provider.GetProviderKey()
		return nil, err
	}
	if response != nil {
		response.ExtraFields.Provider = provider.GetProviderKey()
	}
	return response, nil
}

// OCR is not supported by the Apertus provider.
func (provider *ApertusProvider) OCR(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostOCRRequest) (*schemas.BifrostOCRResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.OCRRequest, provider.GetProviderKey())
}

// BatchDelete is not supported by the Apertus provider.
func (provider *ApertusProvider) BatchDelete(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchDeleteRequest) (*schemas.BifrostBatchDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchDeleteRequest, provider.GetProviderKey())
}

// Passthrough is not supported by the Apertus provider.
func (provider *ApertusProvider) Passthrough(ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.PassthroughRequest, provider.GetProviderKey())
}

// PassthroughStream is not supported by the Apertus provider.
func (provider *ApertusProvider) PassthroughStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, req *schemas.BifrostPassthroughRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.PassthroughStreamRequest, provider.GetProviderKey())
}
