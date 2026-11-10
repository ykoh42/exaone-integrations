import type { BaseModelConfig } from '@strands-agents/sdk'

/** Configuration for the K-EXAONE Chat Completions API. */
export interface ExaoneModelConfig extends BaseModelConfig {
  /** EXAONE API key; defaults to EXAONE_API_KEY. Never returned by getConfig(). */
  apiKey?: string
  /** API base URL, including /v1. */
  baseURL?: string
  /** Request timeout in milliseconds. */
  timeoutMs?: number
  /** Use a streaming API response; false adapts a non-streaming response into events. */
  stream?: boolean
  /** Request the stream usage chunk. Only sent for streaming requests. */
  includeUsage?: boolean
  topK?: number
  presencePenalty?: number
  reasoningEffort?: 'low' | 'medium' | 'high' | 'max'
  preserveThinking?: boolean
  /** Empirically observed option; the official documentation currently disagrees. */
  enableThinking?: boolean
  /** Additional API request parameters. Provider-managed fields cannot be overridden. */
  params?: Record<string, unknown>
  /** Optional fetch implementation for testing or a custom transport. */
  fetch?: typeof globalThis.fetch
  /** Per-request metadata callback; never called with the API key. */
  onMetadata?: (metadata: ExaoneMetadata) => void
}

export interface ExaoneMetadata {
  requestId?: string
  /** Raw API usage fields, including reasoning_tokens when present. */
  usage?: Record<string, unknown>
  rateLimit?: Record<string, string>
}
