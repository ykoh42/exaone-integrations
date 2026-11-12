import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BindToolsInput } from "@langchain/core/language_models/chat_models";
import { AIMessageChunk, defaultToolCallParser } from "@langchain/core/messages";
import type {
  BaseMessage,
  StandardMessageStructure,
  UsageMetadata,
} from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import type { ChatResult } from "@langchain/core/outputs";
import { convertChunksToEvents } from "@langchain/core/language_models/compat";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { RunnableBinding } from "@langchain/core/runnables";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { ExaoneAPIError } from "./errors.js";
import { convertMessagesToExaone } from "./messages.js";
import { RETRYABLE_STATUSES, retryDelayMs, sleepWithSignal } from "./retry.js";
import type { ChatExaoneCallOptions, ChatExaoneParams } from "./types.js";

const DEFAULT_BASE_URL = "https://api.exaone.ai/v1";
const DEFAULT_MODEL = "k-exaone_v2.5";
const MAX_CONTEXT_TOKENS = 524_288;

type ExaoneMessageStructure = StandardMessageStructure;

type ChatCompletionResponse = {
  id?: string;
  model?: string;
  created?: number;
  choices?: Array<{
    index?: number;
    finish_reason?: string | null;
    matched_stop?: string | null;
    logprobs?: unknown;
    message?: Record<string, unknown>;
  }>;
  usage?: Record<string, unknown>;
  error?: unknown;
  [key: string]: unknown;
};

type ChatCompletionChunk = {
  id?: string;
  model?: string;
  created?: number;
  choices?: Array<{
    index?: number;
    finish_reason?: string | null;
    matched_stop?: string | null;
    delta?: Record<string, unknown>;
  }>;
  usage?: Record<string, unknown>;
  error?: unknown;
  [key: string]: unknown;
};

/** LangChain chat model for the text-only EXAONE Chat Completions API. */
export class ChatExaone extends BaseChatModel<
  ChatExaoneCallOptions,
  AIMessageChunk<ExaoneMessageStructure>
> {
  readonly model: string;
  readonly baseUrl: string;
  readonly timeout: number | undefined;
  readonly streamUsage: boolean;

  readonly #apiKey: string;
  private readonly defaultParams: Record<string, unknown>;
  private readonly modelKwargs: Record<string, unknown>;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly retryCount: number;
  private readonly baseRetryDelayMs: number;
  private readonly maxRetryDelayMs: number;

  constructor(fields: ChatExaoneParams = {}) {
    const {
      apiKey,
      baseUrl = DEFAULT_BASE_URL,
      model = DEFAULT_MODEL,
      temperature,
      maxTokens,
      topP,
      topK,
      frequencyPenalty,
      presencePenalty,
      seed,
      reasoningEffort,
      preserveThinking = true,
      enableThinking,
      timeout,
      maxRetries = 2,
      baseRetryDelayMs = 1_000,
      maxRetryDelayMs = 60_000,
      streamUsage = true,
      modelKwargs = {},
      fetch: fetcher = globalThis.fetch,
      ...baseParams
    } = fields;

    super({ ...baseParams, maxRetries: 0 });

    const resolvedApiKey = resolveApiKey(apiKey);
    validateOptions({ maxRetries, baseRetryDelayMs, maxRetryDelayMs, timeout, model, maxTokens });

    this.#apiKey = resolvedApiKey;
    this.model = model;
    this.baseUrl = normalizeBaseUrl(baseUrl);
    this.timeout = timeout;
    this.streamUsage = streamUsage;
    this.fetcher = fetcher;
    this.retryCount = maxRetries;
    this.baseRetryDelayMs = baseRetryDelayMs;
    this.maxRetryDelayMs = maxRetryDelayMs;
    this.modelKwargs = { ...modelKwargs };
    this.defaultParams = {
      ...(temperature === undefined ? {} : { temperature }),
      ...(maxTokens === undefined ? {} : { max_tokens: maxTokens }),
      ...(topP === undefined ? {} : { top_p: topP }),
      ...(topK === undefined ? {} : { top_k: topK }),
      ...(frequencyPenalty === undefined ? {} : { frequency_penalty: frequencyPenalty }),
      ...(presencePenalty === undefined ? {} : { presence_penalty: presencePenalty }),
      ...(seed === undefined ? {} : { seed }),
      ...(reasoningEffort === undefined ? {} : { reasoning_effort: reasoningEffort }),
      chat_template_kwargs: {
        preserve_thinking: preserveThinking,
        ...(enableThinking === undefined ? {} : { enable_thinking: enableThinking }),
      },
    };
  }

  override _llmType(): string {
    return "exaone";
  }

  override _modelType(): string {
    return "chat";
  }

  override _identifyingParams(): Record<string, unknown> {
    return { model: this.model, baseUrl: this.baseUrl, ...this.defaultParams, ...this.modelKwargs };
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatExaoneCallOptions>
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk<ExaoneMessageStructure>,
    ChatExaoneCallOptions
  > {
    const formattedTools = tools.map((tool) => convertToOpenAITool(tool));
    return new RunnableBinding({
      bound: this,
      kwargs: { ...kwargs, tools: formattedTools },
      config: {},
    });
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    const body = this.requestBody(messages, options, false);
    const response = await this.request(body, options.signal, options.maxRetries ?? this.retryCount);
    const payload = await readJsonResponse<ChatCompletionResponse>(response);
    if (payload.error !== undefined && payload.error !== null) {
      throw new ExaoneAPIError(response.status, safeHeaders(response.headers), payload, this.#apiKey);
    }
    const choices = payload.choices;
    if (!Array.isArray(choices) || choices.length !== 1) {
      throw new Error("EXAONE Chat Completions must return exactly one choice for LangChain.");
    }
    const choice = choices[0];
    if (!choice || !isRecord(choice.message)) {
      throw new Error("EXAONE Chat Completions returned a choice without an assistant message.");
    }
    const message = createAssistantMessage(choice.message, payload, choice, response.headers);
    const usage = mapUsage(payload.usage);
    if (usage) message.usage_metadata = usage;
    const generationInfo = responseMetadata(payload, choice, response.headers);
    return {
      generations: [{
        text: typeof message.content === "string" ? message.content : "",
        message,
        generationInfo,
      }],
      ...(usage
        ? {
            llmOutput: {
              tokenUsage: {
                promptTokens: usage.input_tokens,
                completionTokens: usage.output_tokens,
                totalTokens: usage.total_tokens,
              },
            },
          }
        : {}),
    };
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: Parameters<BaseChatModel<ChatExaoneCallOptions>["_streamResponseChunks"]>[2]
  ): AsyncGenerator<ChatGenerationChunk> {
    const body = this.requestBody(messages, options, true);
    const response = await this.request(body, options.signal, options.maxRetries ?? this.retryCount);
    if (!response.body) throw new Error("EXAONE returned an empty streaming response.");

    for await (const payload of parseSse<ChatCompletionChunk>(response.body, options.signal)) {
      for (const generationChunk of streamResponseChunks(payload, response, this.#apiKey)) {
        yield generationChunk;
        if (generationChunk.text) {
          await runManager?.handleLLMNewToken(
            generationChunk.text,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk: generationChunk }
          );
        }
      }
    }
  }

  override async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: Parameters<BaseChatModel<ChatExaoneCallOptions>["_streamChatModelEvents"]>[2]
  ): AsyncGenerator<ChatModelStreamEvent> {
    yield* convertChunksToEvents(
      exposeReasoningContentBlocks(this._streamResponseChunks(messages, options, runManager)),
      options.signal === undefined ? {} : { signal: options.signal }
    );
  }

  private requestBody(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    streaming: boolean
  ): Record<string, unknown> {
    const modelKwargs = {
      ...this.modelKwargs,
      ...(options.modelKwargs ?? {}),
    };
    const body: Record<string, unknown> = {
      ...modelKwargs,
      ...this.defaultParams,
      ...omitUndefined({
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        top_p: options.top_p,
        top_k: options.top_k,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        seed: options.seed,
        reasoning_effort: options.reasoning_effort,
        chat_template_kwargs: options.chat_template_kwargs,
        response_format: options.response_format,
        parallel_tool_calls: options.parallel_tool_calls,
        logprobs: options.logprobs,
        top_logprobs: options.top_logprobs,
        user: options.user,
        tools: options.tools?.map((tool) => convertToOpenAITool(tool)),
        tool_choice: options.tool_choice === undefined ? undefined : formatToolChoice(options.tool_choice),
        stop: options.stop,
      }),
      model: this.model,
      messages: convertMessagesToExaone(messages),
      stream: streaming,
    };

    const defaultTemplate = this.defaultParams.chat_template_kwargs;
    body.chat_template_kwargs = {
      ...(isRecord(modelKwargs.chat_template_kwargs) ? modelKwargs.chat_template_kwargs : {}),
      ...(isRecord(defaultTemplate) ? defaultTemplate : {}),
      ...(isRecord(options.chat_template_kwargs) ? options.chat_template_kwargs : {}),
      preserve_thinking: isRecord(defaultTemplate) ? defaultTemplate.preserve_thinking : true,
    };

    if (streaming) {
      body.stream_options = buildStreamOptions(modelKwargs, options.stream_options, this.streamUsage);
    } else {
      delete body.stream_options;
    }
    validateRequestBody(body, this.model);
    return body;
  }

  private async request(body: Record<string, unknown>, signal: AbortSignal | undefined, maxRetries: number): Promise<Response> {
    const url = `${this.baseUrl}/chat/completions`;
    const requestSignal = this.createRequestSignal(signal);
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      const response = await this.sendRequest(url, body, requestSignal, attempt, maxRetries);
      if (!response) continue;
      if (response.ok) return response;
      await this.retryResponse(response, requestSignal, attempt, maxRetries);
    }
    throw new Error("EXAONE request retry loop ended unexpectedly.");
  }

  private async sendRequest(
    url: string,
    body: Record<string, unknown>,
    signal: AbortSignal | undefined,
    attempt: number,
    maxRetries: number
  ): Promise<Response | undefined> {
    try {
      return await this.fetcher(url, {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.#apiKey}`,
          "content-type": "application/json",
          accept: body.stream === true ? "text/event-stream" : "application/json",
        },
        body: JSON.stringify(body),
        ...(signal === undefined ? {} : { signal }),
      });
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) throw signal?.reason ?? error;
      if (attempt >= maxRetries) throw error;
      const delay = retryDelayMs(new Headers(), attempt, this.baseRetryDelayMs, this.maxRetryDelayMs);
      if (delay === null) throw error;
      await sleepWithSignal(delay, signal);
      return undefined;
    }
  }

  private async retryResponse(
    response: Response,
    signal: AbortSignal | undefined,
    attempt: number,
    maxRetries: number
  ): Promise<void> {
    const payload = await readErrorBody(response);
    const error = new ExaoneAPIError(response.status, safeHeaders(response.headers), payload, this.#apiKey);
    if (!RETRYABLE_STATUSES.has(response.status) || attempt >= maxRetries) throw error;
    const delay = retryDelayMs(
      response.headers,
      attempt,
      this.baseRetryDelayMs,
      this.maxRetryDelayMs,
      response.status === 429 ? Math.max(1_000, this.baseRetryDelayMs) : 0
    );
    if (delay === null) throw error;
    await sleepWithSignal(delay, signal);
  }

  private createRequestSignal(signal?: AbortSignal): AbortSignal | undefined {
    if (this.timeout === undefined) return signal;
    const timeoutSignal = AbortSignal.timeout(this.timeout);
    return signal === undefined ? timeoutSignal : AbortSignal.any([signal, timeoutSignal]);
  }
}

function* streamResponseChunks(
  payload: ChatCompletionChunk,
  response: Response,
  apiKey: string
): Generator<ChatGenerationChunk> {
  if (payload.error !== undefined && payload.error !== null) {
    throw new ExaoneAPIError(response.status, safeHeaders(response.headers), payload, apiKey);
  }
  const usageChunk = createUsageChunk(payload, response.headers);
  if (usageChunk) yield usageChunk;

  const choice = getStreamChoice(payload.choices);
  if (choice) yield createStreamGenerationChunk(payload, choice, response.headers);
}

async function* exposeReasoningContentBlocks(
  chunks: AsyncIterable<ChatGenerationChunk>
): AsyncGenerator<ChatGenerationChunk> {
  const state = { reasoningBlockStarted: false, textBlockStarted: false };

  for await (const chunk of chunks) {
    const message = chunk.message;
    if (!AIMessageChunk.isInstance(message)) {
      yield chunk;
      continue;
    }

    const content = reasoningEventContent(message, state);
    if (content === undefined) {
      yield chunk;
      continue;
    }

    yield chunkWithContentBlocks(chunk, message, content);
  }
}

type ReasoningEventBlock =
  | { type: "reasoning"; reasoning: string; index: number }
  | { type: "text"; text: string; index: number };

function reasoningEventContent(
  message: AIMessageChunk<ExaoneMessageStructure>,
  state: { reasoningBlockStarted: boolean; textBlockStarted: boolean }
): ReasoningEventBlock[] | undefined {
  const reasoning = message.additional_kwargs.reasoning_content;
  const text = typeof message.content === "string" ? message.content : "";
  const hasReasoning = typeof reasoning === "string" && reasoning.length > 0;
  if (!hasReasoning && (!state.reasoningBlockStarted || text.length === 0)) return undefined;

  const content: ReasoningEventBlock[] = [];
  if (!state.textBlockStarted && (hasReasoning || text.length > 0)) {
    content.push({ type: "text", text: "", index: 0 });
    state.textBlockStarted = true;
  }
  if (text.length > 0) content.push({ type: "text", text, index: 0 });
  if (hasReasoning) {
    content.push({ type: "reasoning", reasoning, index: 1 });
    state.reasoningBlockStarted = true;
  }
  return content;
}

function chunkWithContentBlocks(
  chunk: ChatGenerationChunk,
  message: AIMessageChunk<ExaoneMessageStructure>,
  content: ReasoningEventBlock[]
): ChatGenerationChunk {
  // LangChain's typed stream protocol supports reasoning blocks, though its standard AI message
  // structure currently types message content as text-only.
  const eventContent = content as unknown as AIMessageChunk<ExaoneMessageStructure>["content"];
  const eventMessage = new AIMessageChunk<ExaoneMessageStructure>(eventContent);
  Object.assign(eventMessage, {
    additional_kwargs: message.additional_kwargs,
    invalid_tool_calls: message.invalid_tool_calls,
    name: message.name,
    id: message.id,
    response_metadata: message.response_metadata,
    tool_call_chunks: message.tool_call_chunks,
  });
  if (message.usage_metadata !== undefined) eventMessage.usage_metadata = message.usage_metadata;

  return new ChatGenerationChunk({
    text: chunk.text,
    message: eventMessage,
    ...(chunk.generationInfo === undefined ? {} : { generationInfo: chunk.generationInfo }),
  });
}

function createUsageChunk(payload: ChatCompletionChunk, headers: Headers): ChatGenerationChunk | undefined {
  if (!payload.usage) return undefined;
  const usage = mapUsage(payload.usage);
  if (!usage) return undefined;
  return new ChatGenerationChunk({
    text: "",
    message: new AIMessageChunk<ExaoneMessageStructure>({
      content: "",
      usage_metadata: usage,
      response_metadata: { usage: payload.usage, headers: safeHeaders(headers) },
    }),
  });
}

function getStreamChoice(
  choices: ChatCompletionChunk["choices"]
): NonNullable<ChatCompletionChunk["choices"]>[number] | undefined {
  const values = choices ?? [];
  if (values.length > 1 || values.some((choice) => choice.index !== undefined && choice.index !== 0)) {
    throw new Error("EXAONE returned multiple streamed choices; LangChain supports one choice per run.");
  }
  return values[0];
}

function createStreamGenerationChunk(
  payload: ChatCompletionChunk,
  choice: NonNullable<ChatCompletionChunk["choices"]>[number],
  headers: Headers
): ChatGenerationChunk {
  const delta = choice.delta ?? {};
  const content = typeof delta.content === "string" ? delta.content : "";
  const refusal = typeof delta.refusal === "string" ? delta.refusal : undefined;
  const visibleContent = content.trim() === "" && refusal ? refusal : content;
  const additional_kwargs = streamAdditionalKwargs(delta);
  const toolCallChunks = streamToolCallChunks(delta);
  const message = new AIMessageChunk<ExaoneMessageStructure>({
    content: visibleContent,
    ...(Object.keys(additional_kwargs).length === 0 ? {} : { additional_kwargs }),
    ...(toolCallChunks.length === 0 ? {} : { tool_call_chunks: toolCallChunks }),
    response_metadata: responseMetadata(payload, choice, headers),
  });
  const generationInfo = streamGenerationInfo(choice);
  return new ChatGenerationChunk({
    text: visibleContent,
    message,
    ...(Object.keys(generationInfo).length === 0 ? {} : { generationInfo }),
  });
}

function streamAdditionalKwargs(delta: Record<string, unknown>): Record<string, unknown> {
  const additional: Record<string, unknown> = {};
  if (typeof delta.reasoning_content === "string") additional.reasoning_content = delta.reasoning_content;
  if (typeof delta.refusal === "string") additional.refusal = delta.refusal;
  if (isRecord(delta.function_call)) additional.function_call = delta.function_call;
  return additional;
}

function streamToolCallChunks(delta: Record<string, unknown>): Array<Record<string, unknown>> {
  const rawToolCalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
  return rawToolCalls.map((raw) => {
    if (!isRecord(raw)) throw new Error("EXAONE streamed an invalid tool-call delta.");
    const fn = isRecord(raw.function) ? raw.function : {};
    return {
      ...(typeof raw.index === "number" ? { index: raw.index } : {}),
      ...(typeof raw.id === "string" ? { id: raw.id } : {}),
      ...(typeof fn.name === "string" ? { name: fn.name } : {}),
      ...(typeof fn.arguments === "string" ? { args: fn.arguments } : {}),
    };
  });
}

function streamGenerationInfo(
  choice: NonNullable<ChatCompletionChunk["choices"]>[number]
): Record<string, unknown> {
  return {
    ...(choice.finish_reason == null ? {} : { finish_reason: choice.finish_reason }),
    ...(choice.matched_stop == null ? {} : { matched_stop: choice.matched_stop }),
  };
}

function formatToolChoice(choice: ChatExaoneCallOptions["tool_choice"]): unknown {
  if (choice === "any") return "required";
  if (typeof choice === "string" && !["auto", "none", "required"].includes(choice)) {
    return { type: "function", function: { name: choice } };
  }
  return choice;
}

function omitUndefined(values: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined));
}

function buildStreamOptions(
  modelKwargs: Record<string, unknown>,
  callOptions: ChatExaoneCallOptions["stream_options"],
  includeUsage: boolean
): Record<string, unknown> {
  const configured = isRecord(modelKwargs.stream_options) ? modelKwargs.stream_options : {};
  const requested = isRecord(callOptions) ? callOptions : {};
  return {
    ...(includeUsage ? { include_usage: true } : {}),
    ...configured,
    ...requested,
  };
}

function validateRequestBody(body: Record<string, unknown>, model: string): void {
  const maxTokens = body.max_tokens;
  if (maxTokens !== undefined) {
    if (typeof maxTokens !== "number" || !Number.isInteger(maxTokens) || maxTokens < 0) {
      throw new RangeError("max_tokens must be a non-negative integer.");
    }
    if (model === DEFAULT_MODEL && maxTokens > MAX_CONTEXT_TOKENS) {
      throw new RangeError(`max_tokens cannot exceed ${MAX_CONTEXT_TOKENS} for ${DEFAULT_MODEL}.`);
    }
  }
  if (body.n !== undefined && body.n !== 1) {
    throw new RangeError("LangChain ChatExaone supports exactly one completion per request (n=1).");
  }
}

function resolveApiKey(apiKey: string | undefined): string {
  const resolved = apiKey ?? getEnvironmentVariable("EXAONE_API_KEY");
  if (!resolved) {
    throw new Error(
      "EXAONE API key is required. Pass apiKey or set EXAONE_API_KEY in the process environment."
    );
  }
  return resolved;
}

function validateOptions(options: {
  maxRetries: number;
  baseRetryDelayMs: number;
  maxRetryDelayMs: number;
  timeout: number | undefined;
  model: string;
  maxTokens: number | undefined;
}): void {
  if (!Number.isInteger(options.maxRetries) || options.maxRetries < 0) {
    throw new RangeError("maxRetries must be a non-negative integer.");
  }
  if (!Number.isFinite(options.baseRetryDelayMs) || options.baseRetryDelayMs < 0) {
    throw new RangeError("baseRetryDelayMs must be finite and non-negative.");
  }
  if (!Number.isFinite(options.maxRetryDelayMs) || options.maxRetryDelayMs < options.baseRetryDelayMs) {
    throw new RangeError("maxRetryDelayMs must be finite and at least baseRetryDelayMs.");
  }
  if (options.timeout !== undefined && (!Number.isFinite(options.timeout) || options.timeout < 0)) {
    throw new RangeError("timeout must be finite and non-negative.");
  }
  if (
    options.model === DEFAULT_MODEL &&
    options.maxTokens !== undefined &&
    options.maxTokens > MAX_CONTEXT_TOKENS
  ) {
    throw new RangeError(`maxTokens cannot exceed ${MAX_CONTEXT_TOKENS} for ${DEFAULT_MODEL}.`);
  }
}

function createAssistantMessage(
  source: Record<string, unknown>,
  response: ChatCompletionResponse,
  choice: NonNullable<ChatCompletionResponse["choices"]>[number],
  headers: Headers
): AIMessageChunk<ExaoneMessageStructure> {
  const refusal = typeof source.refusal === "string" ? source.refusal : undefined;
  const content = typeof source.content === "string" ? source.content : "";
  const visibleContent = content.trim() === "" && refusal ? refusal : content;
  const additional_kwargs: Record<string, unknown> = {};
  if (typeof source.reasoning_content === "string") additional_kwargs.reasoning_content = source.reasoning_content;
  if (refusal !== undefined) additional_kwargs.refusal = refusal;
  if (source.function_call !== undefined) additional_kwargs.function_call = source.function_call;

  const rawToolCalls = Array.isArray(source.tool_calls) ? source.tool_calls : [];
  const [tool_calls, invalid_tool_calls] = defaultToolCallParser(rawToolCalls as Record<string, unknown>[]);
  const metadata = responseMetadata(response, choice, headers);
  return new AIMessageChunk<ExaoneMessageStructure>({
    content: visibleContent,
    ...(Object.keys(additional_kwargs).length === 0 ? {} : { additional_kwargs }),
    ...(tool_calls.length === 0 ? {} : { tool_calls }),
    ...(invalid_tool_calls.length === 0 ? {} : { invalid_tool_calls }),
    response_metadata: metadata,
  });
}

function responseMetadata(
  response: ChatCompletionResponse | ChatCompletionChunk,
  choice: { finish_reason?: string | null; matched_stop?: string | null; logprobs?: unknown } | undefined,
  headers: Headers
): Record<string, unknown> {
  const metadata: Record<string, unknown> = {
    model_provider: "exaone",
    ...(typeof response.model === "string" ? { model_name: response.model } : {}),
    ...safeResponseFields(response),
    headers: safeHeaders(headers),
  };
  if (choice?.finish_reason != null) metadata.finish_reason = choice.finish_reason;
  if (choice?.matched_stop != null) metadata.matched_stop = choice.matched_stop;
  if (choice?.logprobs !== undefined) metadata.logprobs = choice.logprobs;
  return metadata;
}

function safeResponseFields(response: ChatCompletionResponse | ChatCompletionChunk): Record<string, unknown> {
  const fields: Record<string, unknown> = {};
  for (const key of ["id", "model", "created", "object", "metadata", "moderation", "service_tier", "system_fingerprint", "result_code", "description", "usage"]) {
    if (response[key] !== undefined) fields[key] = response[key];
  }
  return fields;
}

function safeHeaders(headers: Headers): Record<string, string> {
  const names = [
    "content-type",
    "retry-after",
    "retry-after-ms",
    "x-request-id",
    "x-second-rate-limit",
    "x-second-rate-limit-remaining",
    "x-minute-rate-limit",
    "x-minute-rate-limit-remaining",
    "x-day-rate-limit",
    "x-day-rate-limit-remaining",
  ];
  return Object.fromEntries(names.flatMap((name) => {
    const value = headers.get(name);
    return value === null ? [] : [[name, value]];
  }));
}

function mapUsage(usage: Record<string, unknown> | undefined): UsageMetadata | undefined {
  if (!usage) return undefined;
  const input = usage.prompt_tokens;
  const output = usage.completion_tokens;
  const total = usage.total_tokens;
  if (![input, output, total].every((value) => typeof value === "number" && Number.isSafeInteger(value) && value >= 0)) {
    return undefined;
  }
  const result: UsageMetadata = {
    input_tokens: input as number,
    output_tokens: output as number,
    total_tokens: total as number,
  };
  const reasoning = isRecord(usage.completion_tokens_details)
    ? usage.completion_tokens_details.reasoning_tokens
    : usage.reasoning_tokens;
  if (typeof reasoning === "number" && Number.isSafeInteger(reasoning) && reasoning >= 0) {
    result.output_token_details = { reasoning };
  }
  return result;
}

async function readJsonResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!text) throw new Error("EXAONE returned an empty JSON response.");
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error("EXAONE returned malformed JSON.");
  }
}

async function readErrorBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { description: text };
  }
}

async function* parseSse<T>(body: ReadableStream<Uint8Array>, signal?: AbortSignal): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamEnded = false;
  const cancelReader = () => {
    void reader.cancel().catch(() => undefined);
  };
  signal?.addEventListener("abort", cancelReader, { once: true });
  try {
    while (true) {
      signal?.throwIfAborted();
      const { done, value } = await readWithSignal(reader, signal);
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      streamEnded = done;
      buffer = normalizeLineEndings(buffer);
      const frames = takeSseFrames(buffer);
      buffer = frames.remaining;
      const parsed = parseSseFrames<T>(frames.complete);
      yield* parsed.events;
      if (parsed.done) return;
      if (streamEnded) {
        const finalPayload = ssePayload(buffer);
        if (finalPayload && finalPayload !== "[DONE]") yield JSON.parse(finalPayload) as T;
        return;
      }
    }
  } finally {
    signal?.removeEventListener("abort", cancelReader);
    if (!streamEnded) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

function parseSseFrames<T>(frames: string[]): { events: T[]; done: boolean } {
  const events: T[] = [];
  for (const frame of frames) {
    const payload = ssePayload(frame);
    if (payload === "[DONE]") return { events, done: true };
    if (payload) events.push(JSON.parse(payload) as T);
  }
  return { events, done: false };
}

function normalizeLineEndings(buffer: string): string {
  return buffer.replace(/\r\n/g, "\n").replace(/\r(?!$)/g, "\n");
}

function takeSseFrames(buffer: string): { complete: string[]; remaining: string } {
  const complete: string[] = [];
  let separator = buffer.indexOf("\n\n");
  while (separator >= 0) {
    complete.push(buffer.slice(0, separator));
    buffer = buffer.slice(separator + 2);
    separator = buffer.indexOf("\n\n");
  }
  return { complete, remaining: buffer };
}

function ssePayload(frame: string): string {
  return frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
}

async function readWithSignal(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal?: AbortSignal
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (!signal) return reader.read();
  signal.throwIfAborted();
  let onAbort: () => void = () => {};
  const aborted = new Promise<never>((_resolve, reject) => {
    onAbort = () => reject(signal.reason ?? new DOMException("The operation was aborted.", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    return await Promise.race([reader.read(), aborted]);
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}

function normalizeBaseUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new TypeError("EXAONE baseUrl must be an absolute URL.");
  }
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    throw new TypeError("EXAONE baseUrl must use HTTPS, except for loopback HTTP test servers.");
  }
  if (url.username || url.password) throw new TypeError("EXAONE baseUrl must not include URL credentials.");
  if (url.search || url.hash) throw new TypeError("EXAONE baseUrl must not include a query or fragment.");
  return value.replace(/\/+$/, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isAbortError(value: unknown): boolean {
  return value instanceof Error && value.name === "AbortError";
}
