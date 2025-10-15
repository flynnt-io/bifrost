<!-- The pattern we follow here is to keep the changelog for the latest version -->
<!-- Old changelogs are automatically attached to the GitHub releases -->

We're excited to ship v1.3.0 with major quality, compatibility, and governance upgrades across OSS and Enterprise. 

🌟 Highlights
- OTel traces support (OSS): First-class support for OTLP collectors.  
- Responses API (OSS): First-class support for the OpenAI-style Responses format, streaming + non-streaming.
- Drop-in for LiteLLM (OSS): Config-level fallbacks to ease migrations.
- Guardrails (Enterprise): Initial set with AWS Bedrock, Azure Content Moderator, and Patronus AI.
- Provisioning (Enterprise): Okta SCIM now supported alongside Microsoft Entra.
- Adaptive LB Dashboard (Enterprise, beta): Live traffic, weight shifts, and failover visibility.

### Features
- Added Anthropic thinking parameter in Responses API.
- Added Anthropic text completion integration support.
- Latency metrics for all request types now returned in extra (includes inter-token latency for streaming).
- TokenInterceptor interface added to plugins.
- Raw provider response saved in logs (framework v1.1.4).

### Fixes

- Removed extra fields erroneously sent in streaming responses.
- Anthropic tool results aggregation corrected (core v1.2.4).
- String input support fixed for Responses requests.
- Specific timeout error handling across all providers for context.Canceled, context.DeadlineExceeded, and fasthttp.ErrTimeout.
- Pricing manager fixes.

### Improvements

- CORS wildcard matching improved to support domain patterns like *.example.com.

## Closed  tickets

- [#605: [Bug]: UI Docker building errors](https://github.com/maximhq/bifrost/issues/605)
- [#597: [Bug Report] Bedrock streaming has many missing chunks](https://github.com/maximhq/bifrost/issues/597)
- [#567: Handling reasoning content](https://github.com/maximhq/bifrost/issues/567)
- [#565: The "pricing not found for model ..." message is repeated for each request processed, which is too noisy for the warn level.](https://github.com/maximhq/bifrost/issues/565)
- [#552: [Bug]: "index" not specified for tool calls in OpenAI chunks](https://github.com/maximhq/bifrost/issues/552)
- [#543: [Bug]: Indicate timeouts in error response while logging](https://github.com/maximhq/bifrost/issues/543)
- [#542: [Feature]: Logs should show timestamps in browser timezone](https://github.com/maximhq/bifrost/issues/542)
- [#520: [Bug]: tokens and cost for "Chat Stream" requests is missing in logs](https://github.com/maximhq/bifrost/issues/520)
- [#516: [Bug]: Can't delete custom provider from Web UI](https://github.com/maximhq/bifrost/issues/516)
- [#504: [Bug]: cannot use self-hosted SGLang instance with http:// URLs only](https://github.com/maximhq/bifrost/issues/504)
- [#497: [Feature]: Add full support for standard OpenTelemetry GenAI Observability](https://github.com/maximhq/bifrost/issues/497)
- [#479: [Feature]: Support for API Key Authentication in Bedrock](https://github.com/maximhq/bifrost/issues/479)
- [#463: [Feature]: Support for Thinking blocks](https://github.com/maximhq/bifrost/issues/463)
- [#456: [Docs]: Update API reference docs](https://github.com/maximhq/bifrost/issues/456)
- [#451: [Feature]: Offline usage](https://github.com/maximhq/bifrost/issues/451)