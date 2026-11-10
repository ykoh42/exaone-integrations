import type { BaseModelConfig } from '@strands-agents/sdk'

/** Configuration for the K-EXAONE Chat Completions API. */
export interface ExaoneModelConfig extends BaseModelConfig {
  /** EXAONE API key; defaults to EXAONE_API_KEY. Never returned by getConfig(). */
  apiKey?: string
  /** API base URL, including /v1. */
  baseURL?: string
  /** Total request timeout in milliseconds, including retry attempts, delays, and stream consumption. */
  timeoutMs?: number
  /**
   * Additional attempts for retryable HTTP responses before events are exposed.
   * Defaults to 0. Fetch failures are not retried because the request may have been sent.
   */
  maxRetries?: number
  /** Initial exponential backoff window in milliseconds; defaults to 500. */
  baseRetryDelayMs?: number
  /** Maximum permitted delay, including server retry headers; defaults to 30000. */
  maxRetryDelayMs?: number
  /** Use a streaming API response; false adapts a non-streaming response into events. */
  stream?: boolean
  /** Request the stream usage chunk. Only sent for streaming requests. */
  includeUsage?: boolean
  /** Limits token sampling to the top K candidates; -1 disables the filter. */
  topK?: number
  /** Adjusts how strongly repeated topics are penalized. */
  presencePenalty?: number
  /** Controls the EXAONE reasoning effort. */
  reasoningEffort?: 'low' | 'medium' | 'high' | 'max'
  /**
   * Ask EXAONE to retain earlier assistant reasoning in model input. Reasoning
   * present in conversation history is forwarded independently of this setting.
   */
  preserveThinking?: boolean
  /**
   * Empirically observed option; the official documentation currently disagrees.
   * Setting this to false does not guarantee that content contains no reasoning.
   */
  enableThinking?: boolean
  /** Additional API request parameters. Provider-managed fields cannot be overridden. */
  params?: Record<string, unknown>
  /** Optional fetch implementation for testing or a custom transport. */
  fetch?: typeof globalThis.fetch
  /**
   * Called once after a successful response completes. The API key is redacted; response
   * metadata may still contain sensitive content such as logprob tokens, refusal text,
   * or function arguments. Failed and interrupted streams do not invoke this callback.
   */
  onMetadata?: (metadata: ExaoneMetadata) => void
}

/**
 * Provider-specific metadata from a completed EXAONE response.
 *
 * @remarks
 * Some fields can contain sensitive text or encoded data even after the API key is redacted.
 */
export interface ExaoneMetadata {
  /** Request ID from the response headers, when present. */
  requestId?: string
  /** Raw API usage fields, including reasoning_tokens when present. */
  usage?: Record<string, unknown>
  /** Selected rate-limit and retry response headers. */
  rateLimit?: Record<string, string>
  /**
   * Response-level and choice fields, including selected message metadata such as
   * refusal, annotations, audio, and function_call. Excludes content, reasoning and tool_calls.
   * Collected metadata is not size-limited and can grow with the response.
   */
  response?: Record<string, unknown>
}
