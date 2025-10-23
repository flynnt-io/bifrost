# Apertus Provider Implementation

## Overview

The Apertus provider is a custom provider implementation for Bifrost that extends OpenAI-compatible APIs with per-key custom endpoint URLs. It allows each API key to use a different base URL while maintaining full compatibility with the OpenAI API specification.

## Features

- **Per-Key Custom Endpoints**: Each API key can specify its own base URL
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
- **Changes**: Added `ApertusKeyConfig` struct with optional `Endpoint` field

```go
type ApertusKeyConfig struct {
    Endpoint string `json:"endpoint,omitempty"` // Custom endpoint URL for this key
}
```

#### Provider Implementation
- **File**: `core/providers/apertus.go`
- **Key Method**: `getBaseURL(key schemas.Key) string`
  - Returns `key.ApertusKeyConfig.Endpoint` if set
  - Falls back to `provider.networkConfig.BaseURL` if not set

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
export const apertusKeyConfigSchema = z.object({
    endpoint: z.union([
        z.url("Must be a valid URL"),
        z.string().length(0)
    ]).optional(),
});
```

#### UI Form
- **File**: `ui/app/providers/fragments/apiKeysFormFragment.tsx`
- **Field**: Custom Endpoint (Optional)
- **Supports**:
  - Direct URLs: `https://your-api.example.com`
  - Environment variables: `env.APERTUS_ENDPOINT`

## Configuration

### Database Schema

The Apertus endpoint is stored in the `config_keys` table. The migration runs automatically on server startup.

**Migration**: `add_apertus_endpoint_column` (defined in `framework/configstore/migrations.go`)

```sql
ALTER TABLE config_keys ADD COLUMN apertus_endpoint TEXT;
```

The migration system will automatically:
1. Check if the `apertus_endpoint` column exists
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
        "endpoint": "https://custom-endpoint-1.example.com"
      }
    },
    {
      "id": "key-2",
      "name": "Secondary Key",
      "value": "sk-yyyyy",
      "models": ["gpt-4"],
      "weight": 0.5
      // No custom endpoint - will use base_url
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

## Implementation Details

### Provider Delegation

The Apertus provider delegates all request handling to the OpenAI provider's shared handler functions, only changing the URL:

```go
func (provider *ApertusProvider) ChatCompletion(ctx context.Context, key schemas.Key,
    request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {

    return handleOpenAIChatCompletionRequest(
        ctx,
        provider.client,
        provider.getBaseURL(key)+"/v1/chat/completions", // Custom URL per key
        request,
        key,
        provider.networkConfig.ExtraHeaders,
        provider.sendBackRawResponse,
        provider.GetProviderKey(),
        provider.logger,
    )
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
- [ ] Provider saves successfully with custom endpoint
- [ ] Provider saves successfully without custom endpoint (fallback)
- [ ] Chat completion requests use correct endpoint
- [ ] Streaming requests use correct endpoint
- [ ] Embeddings requests use correct endpoint
- [ ] Error responses maintain provider name "apertus"
- [ ] Load balancing works with multiple keys
- [ ] Environment variables work for custom endpoint (e.g., `env.APERTUS_ENDPOINT`)

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
1. Database migration applied (column `apertus_endpoint` exists)
2. `BeforeSave()` method serializing config
3. `AfterFind()` method deserializing config

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
- `core/schemas/account.go` - Added `ApertusKeyConfig` struct
- `core/schemas/bifrost.go` - Added Apertus constant and registration
- `core/providers/apertus.go` - **New provider implementation** (329 lines)
- `core/bifrost.go` - Added factory case for Apertus
- `framework/configstore/tables/key.go` - Added database fields and serialization
- `framework/configstore/migrations.go` - **Added migration for apertus_endpoint column**

### Frontend (TypeScript/React)
- `ui/lib/types/schemas.ts` - Added TypeScript schema validation
- `ui/app/providers/fragments/apiKeysFormFragment.tsx` - Added UI form fields
- `ui/lib/constants/config.ts` - Added model placeholders and key requirements
- `ui/lib/constants/logs.ts` - Added provider name and label

### Total Changes
- **1 new file** (`core/providers/apertus.go`)
- **9 files modified**
- **~400 lines of code added**

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
