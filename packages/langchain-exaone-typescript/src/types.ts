import type {
  BaseChatModelCallOptions,
  BaseChatModelParams,
  BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { ToolDefinition } from "@langchain/core/language_models/base";

export type ExaoneReasoningEffort = "low" | "medium" | "high" | "max";

export interface ChatExaoneParams extends BaseChatModelParams {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  seed?: number;
  reasoningEffort?: ExaoneReasoningEffort;
  preserveThinking?: boolean;
  enableThinking?: boolean;
  timeout?: number;
  maxRetries?: number;
  baseRetryDelayMs?: number;
  maxRetryDelayMs?: number;
  streamUsage?: boolean;
  modelKwargs?: Record<string, unknown>;
  fetch?: typeof globalThis.fetch;
}

export interface ChatExaoneCallOptions extends BaseChatModelCallOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  reasoning_effort?: ExaoneReasoningEffort;
  chat_template_kwargs?: Record<string, unknown>;
  stream_options?: Record<string, unknown> & { include_usage?: boolean };
  response_format?: Record<string, unknown>;
  tools?: BindToolsInput[] | ToolDefinition[];
  parallel_tool_calls?: boolean;
  logprobs?: boolean;
  top_logprobs?: number;
  user?: string;
  modelKwargs?: Record<string, unknown>;
}
