# Apertus Provider Implementation

## Overview

The Apertus provider is a custom provider implementation for Bifrost that extends OpenAI-compatible APIs with per-key custom endpoint URLs. It allows each API key to use a different base URL while maintaining full compatibility with the OpenAI API specification.

## Features

- **Per-Key Custom Endpoints**: Each API key can specify its own base URL
- **Model Name Mappings**: Map user-facing model names to backend-specific model names
- **OpenAI Compatible**: Uses standard OpenAI paths and request/response formats
- **Automatic Fallback**: If no custom endpoint is configured, uses the provider's default base URL
- **Full API Support**: Supports all request types:
  - Text Completion (streaming and non-streaming)
  - Chat Completion (streaming and non-streaming)
  - Responses (streaming and non-streaming)
  - Embeddings
  - Speech Synthesis (streaming and non-streaming)
  - Audio Transcription (streaming and non-streaming)

## Architecture

### Backend (Go)

#### Core Schema
- **File**: `core/schemas/account.go`
- **Changes**: Added `ApertusKeyConfig` struct with optional `Endpoint` and `ModelNameMappings` fields

```go
type ApertusKeyConfig struct {
    Endpoint          string            `json:"endpoint,omitempty"`            // Custom endpoint URL for this key
    ModelNameMappings map[string]string `json:"model_name_mappings,omitempty"` // Mapping of user-facing model names to backend model names
}
```

#### Provider Implementation
- **File**: `core/providers/apertus/apertus.go`
- **Key Methods**:
  - `getBaseURL(key schemas.Key) string`
    - Returns `key.ApertusKeyConfig.Endpoint` if set
    - Falls back to `provider.networkConfig.BaseURL` if not set
  - `getModelName(key schemas.Key, userModel string) string`
    - Returns mapped model name if mapping exists in `key.ApertusKeyConfig.ModelNameMappings`
    - Falls back to original model name if no mapping exists (no error)

#### URL Construction
All API requests use OpenAI-standard paths:
- Chat: `{baseURL}/v1/chat/completions`
- Embeddings: `{baseURL}/v1/embeddings`
- Speech: `{baseURL}/v1/audio/speech`
- Transcription: `{baseURL}/v1/audio/transcriptions`

Where `{baseURL}` is determined by `getBaseURL()` for each request.

### Frontend (TypeScript/React)

#### Schema Validation
- **File**: `ui/lib/types/schemas.ts`
- **Schema**: `apertusKeyConfigSchema`

```typescript
export const apertusKeyConfigSchema = z
    .object({
        endpoint: z.union([
            z.url("Must be a valid URL"),
            z.string().length(0)
        ]).optional(),
        model_name_mappings: z.union([z.record(z.string(), z.string()), z.string()]).optional(),
    })
    .refine(
        (data) => {
            // Validates that model_name_mappings is either:
            // - A valid JSON object
            // - An environment variable reference (e.g., "env.MODEL_MAPPINGS")
            // - An empty string
            // - Not provided
            if (!data.model_name_mappings) return true;
            if (typeof data.model_name_mappings === "object") return true;
            if (typeof data.model_name_mappings === "string") {
                const trimmed = data.model_name_mappings.trim();
                if (trimmed === "") return true;
                if (trimmed.startsWith("env.")) return true;
                try {
                    const parsed = JSON.parse(trimmed);
                    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed);
                } catch {
                    return false;
                }
            }
            return false;
        },
        {
            message: "Model name mappings must be a valid JSON object or an environment variable reference",
            path: ["model_name_mappings"],
        },
    );
```

#### UI Form
- **File**: `ui/app/workspace/providers/fragments/apiKeysFormFragment.tsx`
- **Fields**:
  - **Custom Endpoint (Optional)**:
    - Direct URLs: `https://your-api.example.com`
    - Environment variables: `env.APERTUS_ENDPOINT`
  - **Model Name Mappings (Optional)**:
    - JSON object mapping user-facing model names to backend model names
    - Example: `{"gpt-4o": "real-backend-model-name", "claude-3-opus": "actual-model-id"}`
    - Environment variables: `env.MODEL_MAPPINGS`

## Configuration

### Database Schema

The Apertus configuration is stored in the `config_keys` table. The migrations run automatically on server startup.

**Migrations** (defined in `framework/configstore/migrations.go`):
1. `add_apertus_endpoint_column`:
   ```sql
   ALTER TABLE config_keys ADD COLUMN apertus_endpoint TEXT;
   ```

2. `add_apertus_model_name_mappings_json_column`:
   ```sql
   ALTER TABLE config_keys ADD COLUMN apertus_model_name_mappings_json TEXT;
   ```

The migration system will automatically:
1. Check if each column exists
2. Add it if it doesn't exist
3. Skip if it already exists

**No manual migration needed** - just restart your Bifrost server with `make dev` after updating the code.

### Adding an Apertus Provider via UI

1. Navigate to **Providers** page
2. Click **Add Provider**
3. Select **apertus** from the provider dropdown
4. Configure provider settings:
   - **Name**: A friendly name for the provider
   - **Base URL** (optional): Default endpoint for all keys
   - **Network Config**: Timeouts, retries, headers
5. Add API Keys:
   - **Name**: Key identifier
   - **API Key**: The actual API key value
   - **Models**: List of models this key can access
   - **Weight**: Load balancing weight (0.1-1.0)
   - **Custom Endpoint** (optional): Override base URL for this specific key
   - **Model Name Mappings** (optional): JSON object mapping user-facing model names to backend model names

### Adding an Apertus Provider via API

```bash
POST /api/providers
Content-Type: application/json

{
  "provider": "apertus",
  "network_config": {
    "base_url": "https://default-api.example.com",
    "default_request_timeout_in_seconds": 30,
    "max_retries": 3
  },
  "keys": [
    {
      "id": "key-1",
      "name": "Primary Key",
      "value": "sk-xxxxx",
      "models": ["gpt-4", "gpt-3.5-turbo"],
      "weight": 1.0,
      "apertus_key_config": {
        "endpoint": "https://custom-endpoint-1.example.com",
        "model_name_mappings": {
          "gpt-4": "custom-gpt4-deployment",
          "gpt-3.5-turbo": "turbo-v2"
        }
      }
    },
    {
      "id": "key-2",
      "name": "Secondary Key",
      "value": "sk-yyyyy",
      "models": ["gpt-4"],
      "weight": 0.5
      // No custom endpoint or mappings - will use base_url and original model names
    }
  ]
}
```

## Use Cases

### 1. Multi-Tenant API Routing
Route different customers to different OpenAI-compatible endpoints based on their API key:

```go
Key 1 → https://tenant-a.api.example.com
Key 2 → https://tenant-b.api.example.com
Key 3 → https://tenant-c.api.example.com
```

### 2. Environment-Based Routing
Use different endpoints for development, staging, and production:

```go
Dev Key    → https://dev-api.example.com
Staging Key → https://staging-api.example.com
Prod Key   → https://api.example.com
```

### 3. Load Balancing Across Regions
Distribute traffic across multiple regional endpoints:

```go
Key 1 (weight: 0.5) → https://us-east.api.example.com
Key 2 (weight: 0.3) → https://us-west.api.example.com
Key 3 (weight: 0.2) → https://eu-west.api.example.com
```

### 4. Hybrid Cloud Deployment
Mix different OpenAI-compatible services:

```go
Key 1 → https://api.openai.com (Official OpenAI)
Key 2 → https://your-vllm-instance.com (Self-hosted vLLM)
Key 3 → https://your-fastchat-instance.com (Self-hosted FastChat)
```

## Model Name Mappings

### Overview

Model name mappings allow you to decouple the model names your users request from the actual model names used by your backend API. This is particularly useful when:

- Your backend uses different model identifiers than standard OpenAI names
- You want to provide consistent model names across different backends
- You're using custom fine-tuned models with deployment-specific names
- You need to abstract backend changes from your API consumers

### How It Works

When a request comes in:
1. User requests model `"gpt-4o"`
2. Apertus provider looks up `"gpt-4o"` in the `model_name_mappings` for that key
3. If a mapping exists (e.g., `"gpt-4o": "custom-gpt4-deployment"`), the request model is replaced with `"custom-gpt4-deployment"`
4. If no mapping exists, the original model name is used (fallback behavior)
5. The request is sent to the backend with the transformed model name

### Configuration Example

```json
{
  "keys": [
    {
      "name": "production-key",
      "value": "sk-prod-xxxxx",
      "models": ["gpt-4o", "gpt-4o-mini", "claude-3-opus"],
      "apertus_key_config": {
        "endpoint": "https://your-backend.com",
        "model_name_mappings": {
          "gpt-4o": "prod-gpt4o-ft-2024",
          "gpt-4o-mini": "prod-gpt4o-mini-deployment",
          "claude-3-opus": "anthropic-claude-3-opus-v2"
        }
      }
    }
  ]
}
```

### Request Flow Example

**User Request:**
```bash
POST /v1/chat/completions
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Hello!"}],
  "provider": "apertus"
}
```

**Transformed Backend Request:**
```bash
POST https://your-backend.com/v1/chat/completions
{
  "model": "prod-gpt4o-ft-2024",  # Transformed model name
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

### Use Cases for Model Name Mappings

#### 1. Fine-Tuned Model Deployments
Map standard model names to your fine-tuned versions:

```json
{
  "model_name_mappings": {
    "gpt-4": "company-gpt4-finance-ft",
    "gpt-3.5-turbo": "company-gpt35-support-ft"
  }
}
```

#### 2. Version Abstraction
Hide version changes from users:

```json
{
  "model_name_mappings": {
    "gpt-4": "gpt-4-0125-preview",
    "claude-3": "claude-3-5-sonnet-20241022"
  }
}
```

#### 3. Backend-Specific Model IDs
Map to backend-specific identifiers:

```json
{
  "model_name_mappings": {
    "gpt-4o": "openai/gpt-4o",
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
    "llama-3": "meta/llama-3-70b-instruct"
  }
}
```

#### 4. A/B Testing
Route different model versions for testing:

```json
// Key 1 - Control group
{
  "model_name_mappings": {
    "gpt-4": "gpt-4-0613"
  }
}

// Key 2 - Test group
{
  "model_name_mappings": {
    "gpt-4": "gpt-4-1106-preview"
  }
}
```

### Important Notes

- **Fallback Behavior**: If a model name is not found in the mappings, the original name is used (no error)
- **Per-Key Mappings**: Each key can have its own independent mappings
- **Optional Feature**: Model name mappings are completely optional - omit them to use original model names
- **ListModels**: The `/v1/models` endpoint returns **original** model names from config, not mapped names
- **All Request Types**: Mappings apply to all request types (chat, completion, embeddings, speech, transcription)

## Implementation Details

### Provider Delegation

The Apertus provider delegates all request handling to the OpenAI provider's shared handler functions, only changing the URL and model name:

```go
func (provider *ApertusProvider) ChatCompletion(ctx context.Context, key schemas.Key,
    request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {

    // Apply model name mapping
    request.Model = provider.getModelName(key, request.Model)

    return openai.HandleOpenAIChatCompletionRequest(
        ctx,
        provider.client,
        provider.buildRequestURL(ctx, key, "/v1/chat/completions", schemas.ChatCompletionRequest),
        request,
        key,
        provider.networkConfig.ExtraHeaders,
        providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
        provider.GetProviderKey(),
        provider.logger,
    )
}
```

The `getModelName()` helper method provides the mapping logic:

```go
func (provider *ApertusProvider) getModelName(key schemas.Key, userModel string) string {
    if key.ApertusKeyConfig != nil && key.ApertusKeyConfig.ModelNameMappings != nil {
        if backendModel, ok := key.ApertusKeyConfig.ModelNameMappings[userModel]; ok {
            provider.logger.Debug(fmt.Sprintf("Apertus: Mapped model '%s' to '%s'", userModel, backendModel))
            return backendModel
        }
    }
    return userModel  // Fallback to original name
}
```

### Minimal Code Footprint

The implementation is intentionally minimal to:
- **Reduce maintenance burden**: By delegating to OpenAI handlers
- **Ensure consistency**: Same behavior as OpenAI provider
- **Simplify updates**: Changes to OpenAI handlers automatically apply to Apertus
- **Maintain upstream compatibility**: Minimal changes to core codebase

## Testing

### Manual Testing

1. **Create Apertus Provider**:
   ```bash
   curl -X POST http://localhost:8787/api/providers \
     -H "Content-Type: application/json" \
     -d '{
       "provider": "apertus",
       "keys": [{
         "id": "test-key",
         "name": "Test Key",
         "value": "sk-test-key",
         "models": ["gpt-4"],
         "weight": 1.0,
         "apertus_key_config": {
           "endpoint": "https://your-api.example.com"
         }
       }]
     }'
   ```

2. **Test Chat Completion**:
   ```bash
   curl -X POST http://localhost:8787/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-bifrost-xxxxx" \
     -d '{
       "model": "gpt-4",
       "messages": [{"role": "user", "content": "Hello!"}],
       "provider": "apertus"
     }'
   ```

### Validation Checklist

- [ ] Provider appears in UI provider dropdown
- [ ] Custom endpoint field appears in key configuration form
- [ ] Model name mappings field appears in key configuration form
- [ ] Provider saves successfully with custom endpoint
- [ ] Provider saves successfully without custom endpoint (fallback)
- [ ] Provider saves successfully with model name mappings
- [ ] Provider saves successfully without model name mappings (fallback)
- [ ] Chat completion requests use correct endpoint
- [ ] Chat completion requests use mapped model names
- [ ] Streaming requests use correct endpoint and mapped models
- [ ] Embeddings requests use correct endpoint and mapped models
- [ ] Error responses maintain provider name "apertus"
- [ ] Load balancing works with multiple keys
- [ ] Environment variables work for custom endpoint (e.g., `env.APERTUS_ENDPOINT`)
- [ ] Environment variables work for model name mappings (e.g., `env.MODEL_MAPPINGS`)
- [ ] ListModels returns original model names (not mapped names)

## Troubleshooting

### Issue: "unsupported provider: apertus"

**Cause**: Provider not registered in the factory method.

**Solution**: Verify `core/bifrost.go` includes:
```go
case schemas.Apertus:
    return providers.NewApertusProvider(config, bifrost.logger), nil
```

### Issue: Custom endpoint not being used

**Cause**: `ApertusKeyConfig` not properly saved or loaded from database.

**Solution**: Check:
1. Database migrations applied (columns `apertus_endpoint` and `apertus_model_name_mappings_json` exist)
2. `BeforeSave()` method serializing config
3. `AfterFind()` method deserializing config

### Issue: Model name mappings not being applied

**Cause**: Mappings not properly configured or saved.

**Solution**: Check:
1. Database migration `add_apertus_model_name_mappings_json_column` applied
2. JSON format is valid in the UI form
3. Mappings are not empty (`{}` means no mappings)
4. Model name in request exactly matches a key in the mappings (case-sensitive)
5. Check logs for debug message: "Apertus: Mapped model 'X' to 'Y'"

### Issue: 404 errors on requests

**Cause**: Custom endpoint doesn't implement OpenAI-compatible paths.

**Solution**: Ensure your custom endpoint supports:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/audio/speech`
- `/v1/audio/transcriptions`

## Files Modified

### Backend (Go)
- `core/schemas/account.go` - Added `ApertusKeyConfig` struct with `Endpoint` and `ModelNameMappings` fields
- `core/schemas/bifrost.go` - Added Apertus constant and registration
- `core/providers/apertus/apertus.go` - Provider implementation with `getBaseURL()` and `getModelName()` helpers
- `core/bifrost.go` - Added factory case for Apertus
- `framework/configstore/tables/key.go` - Added database fields (`apertus_endpoint`, `apertus_model_name_mappings_json`) and serialization logic
- `framework/configstore/migrations.go` - Added migrations:
  - `add_apertus_endpoint_column`
  - `add_apertus_model_name_mappings_json_column`

### Frontend (TypeScript/React)
- `ui/lib/types/config.ts` - Added `ApertusKeyConfig` interface and `DefaultApertusKeyConfig`
- `ui/lib/types/schemas.ts` - Added Zod validation for endpoint and model name mappings
- `ui/app/workspace/providers/fragments/apiKeysFormFragment.tsx` - Added UI form fields for endpoint and model name mappings
- `ui/lib/constants/config.ts` - Added model placeholders and key requirements
- `ui/lib/constants/logs.ts` - Added provider name and label

### Total Changes
- **1 new file** (`core/providers/apertus/apertus.go`)
- **10 files modified**
- **~600 lines of code added** (including model name mappings feature)

## Future Enhancements

Potential improvements for future iterations:

1. **Per-Request-Type Endpoints**: Allow different endpoints for chat vs embeddings vs speech
2. **URL Templates**: Support dynamic URL construction with variables (e.g., `https://{region}.api.example.com`)
3. **Automatic Failover**: Fallback to alternate endpoints on failure
4. **Health Checks**: Monitor endpoint availability and route traffic accordingly
5. **Metrics**: Track per-endpoint latency and error rates

## Maintenance Notes

### Updating from Upstream

This implementation follows these principles to maintain upstream compatibility:

1. **Isolated Changes**: Apertus-specific code is self-contained
2. **Follow Patterns**: Uses same patterns as Azure provider
3. **Minimal Core Changes**: Only adds, doesn't modify existing provider logic
4. **Standard Interfaces**: Implements standard `Provider` interface

When pulling from upstream:
- New fields in `Provider` interface will need implementation in `apertus.go`
- Changes to OpenAI handlers automatically apply to Apertus
- New providers in `StandardProviders` list won't conflict with Apertus

### Version Compatibility

This implementation is compatible with:
- Bifrost version: Based on commit `f3e74fa`
- Go version: 1.21+
- Node.js version: 18+
- Next.js version: 15.5.3

## Support

For issues or questions about the Apertus provider:
1. Check this documentation first
2. Review the troubleshooting section
3. Examine the code in `core/providers/apertus.go`
4. Compare with Azure provider implementation for similar patterns

## License

This implementation follows the same license as the Bifrost project.
