import type {
  BaseChatModelCallOptions,
  BaseChatModelParams,
  BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { ToolDefinition } from "@langchain/core/language_models/base";
import type { ModelProfile } from "@langchain/core/language_models/profile";

export type ExaoneReasoningEffort = "low" | "medium" | "high" | "max";

/** Configuration for a LangChain chat model that calls the EXAONE API. */
export interface ChatExaoneParams extends BaseChatModelParams {
  /** Replace the default model profile. An empty profile leaves capabilities unknown. */
  profile?: ModelProfile;
  /** EXAONE API key. Defaults to the `EXAONE_API_KEY` environment variable. */
  apiKey?: string;
  /** API base URL, including `/v1`. Remote URLs must use HTTPS. */
  baseUrl?: string;
  /** EXAONE model ID. Defaults to `k-exaone_v2.5`. */
  model?: string;
  /** Sampling temperature. Omit to use the service default. */
  temperature?: number;
  /** Maximum generated tokens, including reasoning tokens. */
  maxTokens?: number;
  /** Nucleus sampling value. */
  topP?: number;
  /** Top-k sampling value. */
  topK?: number;
  /** Frequency penalty. */
  frequencyPenalty?: number;
  /** Presence penalty. */
  presencePenalty?: number;
  /** Seed for deterministic sampling when supported by the endpoint. */
  seed?: number;
  /** EXAONE reasoning effort to request. */
  reasoningEffort?: ExaoneReasoningEffort;
  /** Preserve EXAONE reasoning across turns. Defaults to `true`. */
  preserveThinking?: boolean;
  /** Whether EXAONE should generate reasoning. Omit to use the service default. */
  enableThinking?: boolean;
  /** Request timeout in milliseconds. Omit for no provider-imposed timeout. */
  timeout?: number;
  /** Maximum retries before output is returned. Defaults to `2`. */
  maxRetries?: number;
  /** Base retry delay in milliseconds. Defaults to `1000`. */
  baseRetryDelayMs?: number;
  /** Maximum retry delay in milliseconds. Defaults to `60000`. */
  maxRetryDelayMs?: number;
  /** Include token usage in the final stream event. Defaults to `true`. */
  streamUsage?: boolean;
  /** Additional endpoint-specific request fields. Reserved fields are provider-controlled. */
  modelKwargs?: Record<string, unknown>;
  /** Fetch implementation, useful for custom transports and deterministic tests. */
  fetch?: typeof globalThis.fetch;
}

/** Per-call options accepted by the EXAONE chat model. */
export interface ChatExaoneCallOptions extends BaseChatModelCallOptions {
  /** Sampling temperature. */
  temperature?: number;
  /** Maximum generated tokens, including reasoning tokens. */
  max_tokens?: number;
  /** Nucleus sampling value. */
  top_p?: number;
  /** Top-k sampling value. */
  top_k?: number;
  /** Frequency penalty. */
  frequency_penalty?: number;
  /** Presence penalty. */
  presence_penalty?: number;
  /** Seed for deterministic sampling when supported by the endpoint. */
  seed?: number;
  /** EXAONE reasoning effort to request. */
  reasoning_effort?: ExaoneReasoningEffort;
  /** EXAONE chat-template options. `preserve_thinking` follows the model setting. */
  chat_template_kwargs?: Record<string, unknown>;
  /** Streaming response options. */
  stream_options?: Record<string, unknown> & { include_usage?: boolean };
  /** Chat Completions response format, including JSON-schema configuration. */
  response_format?: Record<string, unknown>;
  /** Function tools to expose to the model. */
  tools?: BindToolsInput[] | ToolDefinition[];
  /** Whether the model may call multiple tools in one response. */
  parallel_tool_calls?: boolean;
  /** Whether to return token log probabilities when supported by the endpoint. */
  logprobs?: boolean;
  /** Number of top token log probabilities to return when supported. */
  top_logprobs?: number;
  /** Optional request-level user identifier. */
  user?: string;
  /** Additional endpoint-specific request fields. Reserved fields are provider-controlled. */
  modelKwargs?: Record<string, unknown>;
}
