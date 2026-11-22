import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BindToolsInput } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, defaultToolCallParser } from "@langchain/core/messages";
import type {
  BaseMessage,
  StandardMessageStructure,
  UsageMetadata,
} from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import type { ChatResult } from "@langchain/core/outputs";
import { convertChunksToEvents } from "@langchain/core/language_models/compat";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { RunnableBinding, RunnableLambda } from "@langchain/core/runnables";
import type { BaseLanguageModelInput, StructuredOutputMethodOptions } from "@langchain/core/language_models/base";
import { assembleStructuredOutputPipeline, createContentParser, createFunctionCallingParser } from "@langchain/core/language_models/structured_output";
import { toJsonSchema } from "@langchain/core/utils/json_schema";
import { getSchemaDescription, isInteropZodSchema } from "@langchain/core/utils/types";
import type { InteropZodType } from "@langchain/core/utils/types";
import { isSerializableSchema } from "@langchain/core/utils/standard_schema";
import type { SerializableSchema } from "@langchain/core/utils/standard_schema";
import type { ModelProfile } from "@langchain/core/language_models/profile";
import type { Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { ExaoneAPIError, redactSecret, redactSecretText, safeTransportError } from "./errors.js";
import { convertMessagesToExaone } from "./messages.js";
import { RETRYABLE_STATUSES, retryDelayMs, sleepWithSignal } from "./retry.js";
import type { ChatExaoneCallOptions, ChatExaoneParams } from "./types.js";
import { getModelProfile } from "./profiles.js";
import { InvalidToolArguments, checkToolCompletion, isToolCompletion, correctionPrompt, namedChoiceUnsupported, recoveryForCall, recoveryJsonDiagnostic } from "./recovery-context.js";
import type { InvalidCallDetails, RecoveryContext } from "./recovery-context.js";

const DEFAULT_BASE_URL = "https://api.exaone.ai/v1";
const DEFAULT_MODEL = "k-exaone_v2.5";
const MAX_CONTEXT_TOKENS = 524_288;

type ExaoneMessageStructure = StandardMessageStructure;

function nativeResponseFormat(method: string, name: string, schema: unknown, strict: boolean | undefined): Record<string, unknown> {
  if (method === "jsonMode") return { type: "json_object" };
  return { type: "json_schema", json_schema: {
    name, schema, ...(strict === undefined ? {} : { strict }),
  } };
}

type ChatCompletionChoiceMetadata = {
  finish_reason?: string | null;
  matched_stop?: number | string | null;
  logprobs?: unknown;
};

type ChatCompletionResponse = {
  id?: string;
  model?: string;
  created?: number;
  choices?: Array<ChatCompletionChoiceMetadata & {
    index?: number;
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
  choices?: Array<ChatCompletionChoiceMetadata & {
    index?: number;
    delta?: Record<string, unknown>;
  }>;
  usage?: Record<string, unknown>;
  error?: unknown;
  [key: string]: unknown;
};

/** LangChain chat model for the text-only EXAONE Chat Completions API.
 *
 * @example
 * ```ts
 * import { ChatExaone } from "langchain-exaone";
 *
 * const model = new ChatExaone();
 * const response = await model.invoke("Hello");
 * ```
 */
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
  private readonly profileOverride: ModelProfile | undefined;

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
      profile,
      ...baseParams
    } = fields;

    super({ ...baseParams, maxRetries: 0 });

    const resolvedApiKey = resolveApiKey(apiKey);
    validateOptions({ maxRetries, baseRetryDelayMs, maxRetryDelayMs, timeout, model, maxTokens, baseUrl });
    if (profile !== undefined && !isRecord(profile)) {
      throw new TypeError("profile must be an object");
    }

    this.#apiKey = resolvedApiKey;
    this.model = model;
    this.profileOverride = profile === undefined ? undefined : structuredClone(profile);
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

  override get profile(): ModelProfile {
    return this.profileOverride === undefined
      ? (this.baseUrl === DEFAULT_BASE_URL ? getModelProfile(this.model) : {})
      : structuredClone(this.profileOverride);
  }

  override _modelType(): string {
    return "chat";
  }

  override invoke(input: BaseLanguageModelInput, options?: Partial<ChatExaoneCallOptions>): Promise<AIMessageChunk<ExaoneMessageStructure>> {
    const recovery = recoveryForCall();
    const response = super.invoke(input, options);
    if (recovery?.expectedTool === undefined) return response;
    const { expectedTool, failedUsage } = recovery;

    return response.then((message) => {
      // SDK cache hits skip the transport validators, and ToolStrategy can consume the call.
      const call = message.tool_calls?.[0];
      if (message.response_metadata.model_provider !== "exaone" || message.invalid_tool_calls?.length ||
          message.tool_calls?.length !== 1 || call?.name !== expectedTool || !call.id || !isRecord(call.args)) {
        throw new Error("EXAONE recovery did not return the original tool call.");
      }
      checkToolCompletion(message.response_metadata.finish_reason, message.additional_kwargs.refusal);
      const { id, name, content, additional_kwargs, tool_call_chunks, usage_metadata } = message;
      const fields = {
        ...(id === undefined ? {} : { id }),
        ...(name === undefined ? {} : { name }),
        content,
        additional_kwargs,
        tool_calls: [call],
        ...(usage_metadata === undefined ? {} : { usage_metadata }),
        response_metadata: {
          ...message.response_metadata,
          exaone_tool_call_recovery: {
            attempts: 1, ...(failedUsage === undefined ? {} : { failed_usage: failedUsage }),
          },
        },
      };
      // Keep retry usage out of the SDK's cached message; it belongs to this invocation.
      const corrected = (AIMessageChunk.isInstance(message)
        ? new AIMessageChunk<ExaoneMessageStructure>({ ...fields, ...(tool_call_chunks === undefined ? {} : { tool_call_chunks }) })
        : new AIMessage<ExaoneMessageStructure>(fields)) as typeof message;
      recovery.validated = true;
      return corrected;
    });
  }

  override _identifyingParams(): Record<string, unknown> {
    const recovery = recoveryForCall();
    return {
      model: this.model, baseUrl: this.baseUrl, ...this.defaultParams, ...this.modelKwargs,
      // A cache hit bypasses the raw response validators in this model.
      ...(recovery ? { _exaone_tool_call_recovery: [2, recovery.expectedTool ?? null, recovery.required ?? false] } : {}),
    };
  }

  override _combineLLMOutput(...outputs: Array<Record<string, unknown> | undefined>): Record<string, unknown> | undefined {
    let total: { promptTokens: number; completionTokens: number; totalTokens: number } | undefined;
    for (const output of outputs) {
      const usage = output?.tokenUsage;
      if (!isRecord(usage)) continue;
      const { promptTokens, completionTokens, totalTokens } = usage;
      if (typeof promptTokens !== "number" || typeof completionTokens !== "number" || typeof totalTokens !== "number") continue;
      total ??= { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
      total.promptTokens += promptTokens;
      total.completionTokens += completionTokens;
      total.totalTokens += totalTokens;
    }
    return total ? { tokenUsage: total } : undefined;
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
    const { strict, ...boundOptions } = kwargs ?? {};
    const strictConfig: Partial<ChatExaoneCallOptions> = strict === undefined ? {} : { strict };
    return new RunnableBinding({
      bound: this,
      kwargs: { ...boundOptions, tools: formattedTools },
      config: strictConfig,
    });
  }

  override withStructuredOutput<RunOutput extends Record<string, unknown> = Record<string, unknown>>(
    outputSchema: InteropZodType<RunOutput> | SerializableSchema<RunOutput> | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<false>
  ): Runnable<BaseLanguageModelInput, RunOutput>;
  override withStructuredOutput<RunOutput extends Record<string, unknown> = Record<string, unknown>>(
    outputSchema: InteropZodType<RunOutput> | SerializableSchema<RunOutput> | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<true>
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;
  override withStructuredOutput<RunOutput extends Record<string, unknown> = Record<string, unknown>>(
    outputSchema: InteropZodType<RunOutput> | SerializableSchema<RunOutput> | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<boolean>
  ): Runnable<BaseLanguageModelInput, RunOutput> | Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }> {
    const method = config?.method ?? "functionCalling";
    if (!["functionCalling", "jsonMode", "jsonSchema"].includes(method)) {
      throw new Error(`Unsupported structured output method: ${method}`);
    }
    if (config?.strict && method === "jsonMode") {
      throw new Error('"strict" is not supported with the "jsonMode" method.');
    }
    const isSchema = isInteropZodSchema(outputSchema) || isSerializableSchema(outputSchema);
    const functionName = !isSchema && typeof outputSchema.name === "string" ? outputSchema.name : (config?.name ?? "extract");
    const parameters = isSchema ? toJsonSchema(outputSchema) : outputSchema;
    if (method !== "functionCalling") {
      const responseFormat = nativeResponseFormat(method, functionName, parameters, config?.strict);
      const model = new RunnableBinding({ bound: this, kwargs: { response_format: responseFormat }, config: {} });
      return assembleStructuredOutputPipeline(model, createContentParser(outputSchema), config?.includeRaw, config?.includeRaw ? "StructuredOutputRunnable" : "StructuredOutput");
    }
    const model = this.bindTools([{
      type: "function",
      function: {
        name: functionName, description: getSchemaDescription(outputSchema) ?? "A function available to call.", parameters,
        ...(config?.strict === undefined ? {} : { strict: config.strict }),
      },
    }], { tool_choice: { type: "function", function: { name: functionName } } });
    const functionParser = createFunctionCallingParser(outputSchema, functionName);
    const parser = RunnableLambda.from(async (input: BaseMessage): Promise<RunOutput> => {
      const result = await functionParser.invoke(input);
      if (result === undefined) throw new Error(`No tool call found with name ${functionName}.`);
      return result as RunOutput;
    });
    return assembleStructuredOutputPipeline(model, parser, config?.includeRaw, config?.includeRaw ? "StructuredOutputRunnable" : "StructuredOutput");
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    const body = this.requestBody(messages, options, false);
    const recovery = recoveryForCall();
    const tools = recovery && recoverableTools(body, recovery);
    const accounting = recovery?.expectedTool === undefined
      ? undefined
      : new CorrectionFailureUsage(recovery.failedUsage, this.#apiKey);
    try {
      const response = await this.request(body, options.signal, options.maxRetries ?? this.retryCount, false, accounting);
      const payload = await readJsonResponse<ChatCompletionResponse>(response, this.#apiKey);
      accounting?.record(payload);
      if (payload.error !== undefined && payload.error !== null) {
        throw new ExaoneAPIError(response.status, safeHeaders(response.headers), payload, this.#apiKey);
      }
      const choices = payload.choices;
      if (!Array.isArray(choices) || choices.length === 0) {
        throw new Error("EXAONE Chat Completions must return at least one choice.");
      }
      options.signal?.throwIfAborted();
      if (recovery?.expectedTool !== undefined) {
        validateReplacement(choices, recovery.expectedTool);
      } else if (tools) validateRecoveryResponse(choices, this.#apiKey, tools, payload.usage);
      const usage = mapUsage(payload.usage);
      const generations = choices.map((choice) => {
        if (!isRecord(choice) || !isRecord(choice.message)) {
          throw new Error("EXAONE Chat Completions returned a choice without an assistant message.");
        }
        const message = createAssistantMessage(choice.message, payload, choice, response.headers);
        if ((options.outputVersion ?? this.outputVersion) === "v1") setStandardContent(message);
        // EXAONE reports request-level usage, shared by all candidates.
        if (usage) message.usage_metadata = usage;
        return {
          text: typeof message.content === "string" ? message.content : "",
          message,
          generationInfo: responseMetadata(payload, choice, response.headers),
        };
      });
      return {
        generations,
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
    } catch (error) {
      accounting?.attach(error, options.signal);
      throw error;
    }
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: Parameters<BaseChatModel<ChatExaoneCallOptions>["_streamResponseChunks"]>[2]
  ): AsyncGenerator<ChatGenerationChunk> {
    const body = this.requestBody(messages, options, true);
    const recovery = recoveryForCall();
    const tools = recovery && recoverableTools(body, recovery);
    const signal = this.createRequestSignal(options.signal);
    const retries = options.maxRetries ?? this.retryCount;
    const response = await this.request(body, signal, retries, true);
    if (!response.body) throw new Error("EXAONE returned an empty streaming response.");

    const seenMetadata = new Map<string, unknown>();
    const stream = parseSse<ChatCompletionChunk>(response.body, signal, this.#apiKey, recovery !== undefined);
    const payloads: AsyncIterable<RecoveryPayload> = recovery && tools
      ? guardRecoveryStream(stream, response, this.#apiKey, tools, recovery, (call) => this.correctStream(body, call, signal, retries))
      : stream;
    for await (const payload of payloads) {
      for (const originalChunk of modelStreamChunks(payload, response, this.#apiKey)) {
        signal?.throwIfAborted();
        deduplicateStreamMetadata(originalChunk.message.response_metadata, seenMetadata);
        const generationChunk = (options.outputVersion ?? this.outputVersion) === "v1"
          ? standardStreamChunk(originalChunk)
          : originalChunk;
        yield generationChunk;
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

  private async correctStream(body: Record<string, unknown>, call: InvalidCallDetails, signal: AbortSignal | undefined, retries: number) {
    const assistant = {
      role: "assistant", content: call.content,
      ...(call.reasoning === undefined ? {} : { reasoning_content: call.reasoning }),
      tool_calls: [{ id: call.id, type: "function", function: { name: call.name, arguments: "{}" } }],
    };
    const feedback = { role: "tool", tool_call_id: call.id, content: correctionPrompt(call) };
    const request: Record<string, unknown> = {
      ...body,
      messages: [...(body.messages as unknown[]), assistant, feedback],
      tool_choice: { type: "function", function: { name: call.name } },
    };
    let response: Response;
    let usage = new CorrectionFailureUsage(call.usage, this.#apiKey);
    try {
      try {
        response = await this.request(request, signal, retries, true, usage);
      } catch (error) {
        if (!namedChoiceUnsupported(error)) throw error;
        signal?.throwIfAborted();
        request.tool_choice = "required";
        usage = new CorrectionFailureUsage(call.usage, this.#apiKey);
        response = await this.request(request, signal, retries, true, usage);
      }
      if (!response.body) throw new Error("EXAONE returned an empty correction stream.");
      const router = new RecoveryPayloadRouter(false);
      for await (const payload of parseSse<ChatCompletionChunk>(response.body, signal, this.#apiKey, true, usage)) {
        for (const early of router.accept(payload)) validateStreamFrame(early, response, this.#apiKey);
      }
      signal?.throwIfAborted();
      const replacement = validateRecoveryPayloads(router.finish(), response, this.#apiKey, new Set(), call.name, router);
      if (!replacement) throw new Error("EXAONE returned no validated correction.");
      return replacement;
    } catch (error) {
      usage.attach(error, signal);
      throw error;
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
        tools: options.tools?.map((tool) => formatTool(tool, options.strict)),
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
    validateRequestBody(body, this.baseUrl);
    return body;
  }

  private async request(
    body: Record<string, unknown>, signal: AbortSignal | undefined, maxRetries: number, hasDeadline = false,
    accounting?: CorrectionFailureUsage,
  ): Promise<Response> {
    const url = `${this.baseUrl}/chat/completions`;
    const requestSignal = hasDeadline ? signal : this.createRequestSignal(signal);
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      const response = await this.sendRequest(url, body, requestSignal, attempt, maxRetries);
      if (!response) continue;
      if (response.ok) return response;
      await this.retryResponse(response, requestSignal, attempt, maxRetries, accounting);
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
      if (signal?.aborted) throw signal.reason;
      if (isAbortError(error)) throw safeTransportError(error, this.#apiKey);
      if (attempt >= maxRetries) throw safeTransportError(error, this.#apiKey);
      const delay = retryDelayMs(new Headers(), attempt, this.baseRetryDelayMs, this.maxRetryDelayMs);
      if (delay === null) throw safeTransportError(error, this.#apiKey);
      await sleepWithSignal(delay, signal);
      return undefined;
    }
  }

  private async retryResponse(
    response: Response,
    signal: AbortSignal | undefined,
    attempt: number,
    maxRetries: number,
    accounting?: CorrectionFailureUsage,
  ): Promise<void> {
    const payload = await readErrorBody(response, this.#apiKey);
    accounting?.record(payload);
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

function* modelStreamChunks(payload: RecoveryPayload, response: Response, key: string): Generator<ChatGenerationChunk> {
  const accounting = payload[correctionAccounting];
  if (accounting === undefined) {
    yield* streamResponseChunks(payload, response, key);
    return;
  }
  yield new ChatGenerationChunk({ text: "", message: new AIMessageChunk<ExaoneMessageStructure>({
    content: "",
    ...(accounting.usage === undefined ? {} : { usage_metadata: accounting.usage }),
    response_metadata: {
      exaone_tool_call_recovery: accounting.metadata,
      ...(accounting.wireUsage ? { usage: accounting.wireUsage } : {}),
    },
  }) });
}

function validateStreamFrame(payload: ChatCompletionChunk, response: Response, key: string): void {
  for (const _chunk of streamResponseChunks(payload, response, key)) { /* validation only */ }
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
  if (choice) {
    const chunk = createStreamGenerationChunk(payload, choice, response.headers);
    if (usageChunk) delete chunk.message.response_metadata.usage;
    yield chunk;
  }
}

function newMetadataFields(value: Record<string, unknown>, seen: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    const previous = seen[key];
    if (isRecord(item) && isRecord(previous)) {
      const nested = newMetadataFields(item, previous);
      if (Object.keys(nested).length) result[key] = nested;
    } else if (!(key in seen) || JSON.stringify(previous) !== JSON.stringify(item)) {
      result[key] = item;
      seen[key] = structuredClone(item);
    }
  }
  return result;
}

function deduplicateStreamMetadata(metadata: Record<string, unknown>, seen: Map<string, unknown>): void {
  for (const key of ["model_name", "model", "created", "object", "metadata", "moderation", "service_tier", "system_fingerprint", "result_code", "description", "headers"]) {
    const value = metadata[key];
    if (value === undefined || value === null) continue;
    const previous = seen.get(key);
    if (key === "metadata" && isRecord(value) && isRecord(previous)) {
      const remaining = newMetadataFields(value, previous);
      if (Object.keys(remaining).length) metadata[key] = remaining;
      else delete metadata[key];
    } else if (seen.has(key) && JSON.stringify(previous) === JSON.stringify(value)) delete metadata[key];
    else seen.set(key, structuredClone(value));
  }
}

function setStandardContent(message: AIMessage<ExaoneMessageStructure> | AIMessageChunk<ExaoneMessageStructure>): void {
  if (message.response_metadata.output_version === "v1") return;
  const blocks: Array<Record<string, unknown>> = [];
  const reasoning = message.additional_kwargs.reasoning_content;
  if (typeof reasoning === "string" && reasoning) blocks.push({ type: "reasoning", reasoning, index: 1 });
  if (typeof message.content === "string" && message.content) blocks.push({ type: "text", text: message.content, index: 0 });
  const refusal = message.additional_kwargs.refusal;
  if (typeof refusal === "string" && refusal) blocks.push({ type: "refusal", refusal, index: 2 });
  if (AIMessageChunk.isInstance(message) && message.tool_call_chunks?.length) {
    for (const call of message.tool_call_chunks) {
      blocks.push({ ...call, type: "tool_call_chunk", index: (call.index ?? 0) + 3 });
    }
  } else {
    for (const [index, call] of (message.tool_calls ?? []).entries()) {
      blocks.push({ ...call, type: "tool_call", index: index + 3 });
    }
  }
  message.content = blocks as unknown as AIMessageChunk<ExaoneMessageStructure>["content"];
  message.response_metadata = { ...message.response_metadata, output_version: "v1" };
}

function standardStreamChunk(chunk: ChatGenerationChunk): ChatGenerationChunk {
  if (AIMessageChunk.isInstance(chunk.message)) setStandardContent(chunk.message);
  return chunk;
}

async function* exposeReasoningContentBlocks(
  chunks: AsyncIterable<ChatGenerationChunk>
): AsyncGenerator<ChatGenerationChunk> {
  const state = { reasoningBlockStarted: false, textBlockStarted: false, refusal: "" };

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
  | { type: "refusal"; refusal: string; index: number }
  | { type: "text"; text: string; index: number };

function reasoningEventContent(
  message: AIMessageChunk<ExaoneMessageStructure>,
  state: { reasoningBlockStarted: boolean; textBlockStarted: boolean; refusal: string }
): ReasoningEventBlock[] | undefined {
  const reasoning = message.additional_kwargs.reasoning_content;
  const text = typeof message.content === "string" ? message.content : message.content
    .filter((part) => typeof part !== "string" && part.type === "text")
    .map((part) => typeof part === "object" && "text" in part && typeof part.text === "string" ? part.text : "")
    .join("");
  const refusal = message.additional_kwargs.refusal;
  const hasReasoning = typeof reasoning === "string" && reasoning.length > 0;
  const hasRefusal = typeof refusal === "string" && refusal.length > 0;
  const hasToolChunks = (message.tool_call_chunks?.length ?? 0) > 0;
  // Tool chunks need stable, separate indices even before reasoning appears. v1 also
  // includes them in content, so rebuild content without duplicating those chunks.
  const hasStandardText = message.response_metadata.output_version === "v1" && text.length > 0;
  if (!hasReasoning && !hasRefusal && !hasToolChunks && !hasStandardText &&
      (!(state.reasoningBlockStarted || state.refusal) || !text)) return undefined;

  const content: ReasoningEventBlock[] = [];
  if (!state.textBlockStarted && (hasReasoning || hasRefusal || text.length > 0)) {
    content.push({ type: "text", text: "", index: 0 });
    state.textBlockStarted = true;
  }
  if (text.length > 0) content.push({ type: "text", text, index: 0 });
  if (hasReasoning) {
    content.push({ type: "reasoning", reasoning, index: 1 });
    state.reasoningBlockStarted = true;
  }
  if (hasRefusal) {
    state.refusal += refusal;
    content.push({ type: "refusal", refusal: state.refusal, index: 2 });
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
    tool_call_chunks: message.tool_call_chunks?.map((call) => ({ ...call, index: (call.index ?? 0) + 3 })),
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
  return new ChatGenerationChunk({
    text: "",
    message: new AIMessageChunk<ExaoneMessageStructure>({
      content: "",
      ...(usage ? { usage_metadata: usage } : {}),
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
  for (const field of ["content", "reasoning_content", "reasoning", "refusal"] as const) {
    if (delta[field] != null && typeof delta[field] !== "string") {
      throw new Error(`EXAONE stream delta ${field} must be text.`);
    }
  }
  const content = typeof delta.content === "string" ? delta.content : "";
  const additional_kwargs = streamAdditionalKwargs(delta);
  const toolCallChunks = streamToolCallChunks(delta);
  const message = new AIMessageChunk<ExaoneMessageStructure>({
    content,
    ...(Object.keys(additional_kwargs).length === 0 ? {} : { additional_kwargs }),
    ...(toolCallChunks.length === 0 ? {} : { tool_call_chunks: toolCallChunks }),
    response_metadata: responseMetadata(payload, choice, headers),
  });
  const generationInfo = streamGenerationInfo(choice);
  return new ChatGenerationChunk({
    text: content,
    message,
    ...(Object.keys(generationInfo).length === 0 ? {} : { generationInfo }),
  });
}

function streamAdditionalKwargs(delta: Record<string, unknown>): Record<string, unknown> {
  const additional: Record<string, unknown> = {};
  const reasoning = responseReasoning(delta, "stream delta");
  if (reasoning !== undefined) additional.reasoning_content = reasoning;
  if (typeof delta.refusal === "string") additional.refusal = delta.refusal;
  if (isRecord(delta.function_call)) additional.function_call = delta.function_call;
  return additional;
}

function streamToolCallChunks(delta: Record<string, unknown>): Array<Record<string, unknown>> {
  const rawToolCalls = toolCallEntries(delta.tool_calls, "stream delta");
  return rawToolCalls.map((raw) => {
    const fn = isRecord(raw.function) ? raw.function : {};
    if ((raw.index != null && (typeof raw.index !== "number" || !Number.isInteger(raw.index))) ||
        (raw.id != null && typeof raw.id !== "string") ||
        (fn.name != null && typeof fn.name !== "string") ||
        (fn.arguments != null && typeof fn.arguments !== "string")) {
      throw new Error("EXAONE stream tool call fields have invalid types.");
    }
    return {
      ...(typeof raw.index === "number" ? { index: raw.index } : {}),
      ...(typeof raw.id === "string" ? { id: raw.id } : {}),
      ...(typeof fn.name === "string" ? { name: fn.name } : {}),
      ...(typeof fn.arguments === "string" ? { args: fn.arguments } : {}),
    };
  });
}

function toolCallEntries(value: unknown, context: "response" | "stream delta"): Record<string, unknown>[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) throw new Error(`EXAONE ${context} tool_calls must be an array.`);
  if (!value.every(isRecord)) {
    const location = context === "response" ? "response" : "stream";
    throw new Error(`EXAONE ${location} tool call entries must be objects.`);
  }
  for (const toolCall of value) {
    validateToolCall(toolCall, context);
  }
  return value;
}

function validateToolCall(toolCall: Record<string, unknown>, context: "response" | "stream delta"): void {
  if (toolCall.type != null && toolCall.type !== "function") {
    throw new Error("EXAONE returned an unsupported tool call type.");
  }
  if (context === "response" && !isRecord(toolCall.function)) {
    throw new Error("EXAONE response tool call function must be an object.");
  }
  if (context === "stream delta" && toolCall.function != null && !isRecord(toolCall.function)) {
    throw new Error("EXAONE stream tool call function must be an object.");
  }
  if (context === "response") validateResponseToolCall(toolCall);
}

function validateResponseToolCall(toolCall: Record<string, unknown>): void {
  const fn = toolCall.function as Record<string, unknown>;
  if (typeof fn.name !== "string" || typeof fn.arguments !== "string" ||
      (toolCall.id != null && typeof toolCall.id !== "string")) {
    throw new Error("EXAONE response tool calls require a name, string arguments, and optional string id.");
  }
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

function formatTool(tool: BindToolsInput, strict: boolean | undefined): ReturnType<typeof convertToOpenAITool> {
  const formatted = convertToOpenAITool(tool);
  if (strict === undefined || formatted.type !== "function") return formatted;
  // The SDK converter mutates preformatted inputs when given strict. Copy both
  // levels so per-call overrides do not alter a user's reusable tool object.
  const functionWithStrict = { ...formatted.function, strict };
  return { ...formatted, function: functionWithStrict };
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

function validateRequestBody(body: Record<string, unknown>, baseUrl: string): void {
  if (typeof body.reasoning_effort === "number" && !Number.isFinite(body.reasoning_effort)) {
    throw new TypeError("reasoning_effort must be finite.");
  }
  const maxTokens = body.max_tokens;
  if (maxTokens !== undefined) {
    if (typeof maxTokens !== "number" || !Number.isInteger(maxTokens) || maxTokens < 0) {
      throw new RangeError("max_tokens must be a non-negative integer.");
    }
    if (body.model === DEFAULT_MODEL && baseUrl === DEFAULT_BASE_URL && maxTokens > MAX_CONTEXT_TOKENS) {
      throw new RangeError(`max_tokens cannot exceed ${MAX_CONTEXT_TOKENS} for ${DEFAULT_MODEL}.`);
    }
  }
  if (body.n !== undefined && (!Number.isInteger(body.n) || (body.n as number) < 1)) {
    throw new RangeError("n must be a positive integer.");
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
  baseUrl: string;
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
  if (options.model === DEFAULT_MODEL && options.baseUrl.replace(/\/+$/, "") === DEFAULT_BASE_URL &&
      options.maxTokens !== undefined && options.maxTokens > MAX_CONTEXT_TOKENS) {
    throw new RangeError(`maxTokens cannot exceed ${MAX_CONTEXT_TOKENS} for ${DEFAULT_MODEL}.`);
  }
}

function responseReasoning(source: Record<string, unknown>, context: string): string | undefined {
  const original = typeof source.reasoning_content === "string" ? source.reasoning_content : undefined;
  const alternate = typeof source.reasoning === "string" ? source.reasoning : undefined;
  if (original && alternate && original !== alternate) {
    throw new Error(`EXAONE ${context} has conflicting reasoning and reasoning_content fields.`);
  }
  return original || alternate || original;
}

function validateAssistantTextFields(source: Record<string, unknown>): void {
  for (const field of ["content", "reasoning_content", "reasoning", "refusal"] as const) {
    if (source[field] != null && typeof source[field] !== "string") {
      throw new Error(`EXAONE assistant ${field} must be a string.`);
    }
  }
}

function validateReplacement(choices: unknown[], name: string): void {
  const choice = choices[0];
  if (choices.length !== 1 || !isRecord(choice) || !isRecord(choice.message) ||
      !isToolCompletion(choice.finish_reason, choice.message.refusal) ||
      !Array.isArray(choice.message.tool_calls) || choice.message.tool_calls.length !== 1) {
    throw new Error("EXAONE recovery must return exactly one completed tool call.");
  }
  const [call] = toolCallEntries(choice.message.tool_calls, "response");
  if (call && typeof call.id === "string" && call.id && isRecord(call.function) &&
      call.function.name === name && typeof call.function.arguments === "string") {
    try {
      if (isRecord(JSON.parse(call.function.arguments))) return;
    } catch { /* Do not propagate parser errors containing the raw input. */ }
  }
  throw new Error("EXAONE recovery must return the original tool with a valid JSON object.");
}

function recoverableTools(body: Record<string, unknown>, recovery: RecoveryContext): ReadonlySet<string> {
  const names = new Set<string>();
  for (const tool of Array.isArray(body.tools) ? body.tools : []) {
    if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === "string") names.add(tool.function.name);
  }
  const choice = body.tool_choice;
  const named = isRecord(choice) && choice.type === "function" && isRecord(choice.function) ? choice.function.name : undefined;
  const expected = recovery.expectedTool;
  if (expected !== undefined && (!names.has(expected) || (recovery.required ? choice !== "required" : named !== expected))) {
    throw new Error("The effective EXAONE request does not allow the tool correction.");
  }
  if (choice == null || choice === "auto" || choice === "required") return names;
  return new Set(typeof named === "string" && names.has(named) ? [named] : []);
}

function validateRecoveryResponse(choices: unknown[], key: string, tools: ReadonlySet<string>, usage?: unknown): void {
  if (choices.length > 1) {
    validateMultipleRecoveryChoices(choices, key, tools, usage);
    return;
  }
  if (choices.length !== 1 || !isRecord(choices[0])) return;
  const choice = choices[0];
  if (!isRecord(choice.message)) return;
  const message = choice.message;
  if (!Array.isArray(message.tool_calls)) return;
  validateAssistantTextFields(message);
  const entries = toolCallEntries(message.tool_calls, "response");
  if (!entries.length) return;
  checkToolCompletion(choice.finish_reason, message.refusal);
  const malformed = entries.flatMap(malformedRecoveryCall);
  if (!malformed.length) return;
  if (malformed.length !== 1 || entries.length !== 1) {
    throw new Error("Cannot recover a response with multiple tool calls.");
  }
  const call = malformed[0]!;
  throwInvalidRecoveryCall(call, message, key, tools, usage);
}

function validateMultipleRecoveryChoices(choices: unknown[], key: string, tools: ReadonlySet<string>, usage: unknown): void {
  for (const choice of choices) {
    try {
      validateRecoveryResponse([choice], key, tools, usage);
    } catch (error) {
      if (error instanceof InvalidToolArguments) throw new Error("Cannot recover multiple completion choices.");
      throw error;
    }
  }
}

interface MalformedRecoveryCall { id: string; name: string; args: string; diagnostic: string }

function malformedRecoveryCall(call: Record<string, unknown>): MalformedRecoveryCall[] {
  if (!isRecord(call.function) || typeof call.function.arguments !== "string") return [];
  // Preserve the SDK's existing treatment of empty arguments (notably for zero-argument tools).
  if (!call.function.arguments.trim()) return [];
  try {
    JSON.parse(call.function.arguments);
    return [];
  } catch (error) {
    if (typeof call.id !== "string" || !call.id || typeof call.function.name !== "string" || !call.function.name) {
      throw new Error("Cannot recover malformed tool arguments without a call ID and name.");
    }
    return [{ id: call.id, name: call.function.name, args: call.function.arguments,
      diagnostic: recoveryJsonDiagnostic(error, call.function.arguments) }];
  }
}

function throwInvalidRecoveryCall(
  call: MalformedRecoveryCall, message: Record<string, unknown>, key: string, tools: ReadonlySet<string>, usage: unknown
): never {
  if ([call.args, call.name, call.id].some((value) => redactSecretText(value, key) !== value)) {
    throw new Error("Cannot recover tool arguments containing credentials.");
  }
  if (!tools.has(call.name)) throw new Error("The effective EXAONE request does not allow the tool correction.");
  if (message.audio != null || message.annotations != null || message.function_call != null) {
    throw new Error("Cannot recover tool calls with unsupported assistant content.");
  }
  const reasoning = responseReasoning(message, "assistant");
  throw new InvalidToolArguments({
    id: call.id,
    name: call.name,
    arguments: call.args,
    diagnostic: call.diagnostic,
    content: typeof message.content === "string" ? message.content : "",
    ...(reasoning === undefined ? {} : { reasoning }),
    ...(isRecord(usage) ? { usage: redactSecret(usage, key) as Record<string, unknown> } : {}),
  });
}

async function* guardRecoveryStream(
  stream: AsyncIterable<ChatCompletionChunk>, response: Response, key: string, tools: ReadonlySet<string>, recovery: RecoveryContext,
  correct: (call: InvalidCallDetails) => Promise<{ arguments: string; usage: unknown }>,
): AsyncGenerator<RecoveryPayload> {
  const router = new RecoveryPayloadRouter(tools.size > 0 && recovery.expectedTool === undefined);
  for await (const payload of stream) yield* router.accept(payload);
  const payloads = router.finish();
  try {
    validateRecoveryPayloads(payloads, response, key, tools, recovery.expectedTool, router);
  } catch (error) {
    if (!(error instanceof InvalidToolArguments)) throw error;
    const replacement = await correct(error.details);
    yield* replaceStreamArguments(payloads, error.details, replacement, key);
    return;
  }
  yield* payloads;
}

function validateRecoveryPayloads(
  payloads: ChatCompletionChunk[], response: Response, key: string, tools: ReadonlySet<string>, expectedTool: string | undefined,
  router: RecoveryPayloadRouter,
): { arguments: string; usage: unknown } | undefined {
  const state: RecoveryStreamState = {
    message: { ...router.unsupportedRecoveryFields, content: router.content.join(""), reasoning_content: router.reasoning.join("") },
    calls: new Map(),
  };
  for (const payload of payloads) {
    validateStreamFrame(payload, response, key);
    accumulateRecoveryPayload(state, payload);
  }
  state.message.tool_calls = [...state.calls.values()];
  if (router.refusal !== undefined) state.message.refusal = router.refusal;
  const choices = [{ finish_reason: router.finishReason, message: state.message }];
  if (expectedTool !== undefined) {
    if (state.multipleIdFragments) throw new Error("EXAONE recovery returned an ambiguous streamed tool call ID.");
    if (state.message.function_call != null) throw new Error("EXAONE recovery returned a legacy function call.");
    validateReplacement(choices, expectedTool);
    return { arguments: state.calls.values().next().value!.function.arguments, usage: state.usage };
  }
  try {
    validateRecoveryResponse(choices, key, tools, state.usage);
  } catch (error) {
    if (error instanceof InvalidToolArguments && state.multipleIdFragments) {
      throw new Error("Cannot recover a tool call with ambiguous streamed ID fragments.");
    }
    throw error;
  }
  return undefined;
}

function* replaceStreamArguments(
  payloads: ChatCompletionChunk[], call: InvalidCallDetails, correction: { arguments: string; usage: unknown }, key: string,
): Generator<RecoveryPayload> {
  let written = false;
  let originalUsage: Record<string, unknown> | undefined;
  for (const payload of payloads) {
    const { usage, ...frame } = payload;
    if (usage) originalUsage = usage;
    const choice = frame.choices?.[0];
    const fragments = choice?.delta?.tool_calls;
    if (choice?.delta && Array.isArray(fragments) && fragments.length) {
      const { tool_calls: _toolCalls, ...delta } = choice.delta;
      if (!written) {
        written = true;
        yield { ...frame, choices: [{ ...choice, delta: { ...delta, tool_calls: [{
          id: call.id, type: "function", index: fragments[0]!.index,
          function: { name: call.name, arguments: correction.arguments },
        }] } }] };
        continue;
      }
      yield { ...frame, choices: [{ ...choice, delta }] };
    } else {
      yield frame;
    }
  }
  const correctedUsage = isRecord(correction.usage) ? correction.usage : undefined;
  yield { choices: [], [correctionAccounting]: {
    usage: mergeRecoveryUsage(mapUsage(originalUsage), mapUsage(correctedUsage)),
    metadata: { attempts: 1, failed_usage: call.usage, correction_usage: redactSecret(correctedUsage, key) },
    wireUsage: redactSecret(originalUsage, key) as Record<string, unknown> | undefined,
  } };
}

function mergeRecoveryUsage(left?: UsageMetadata, right?: UsageMetadata): UsageMetadata | undefined {
  if (!left || !right) return left ?? right;
  const result: UsageMetadata = {
    input_tokens: left.input_tokens + right.input_tokens,
    output_tokens: left.output_tokens + right.output_tokens,
    total_tokens: left.total_tokens + right.total_tokens,
  };
  if (!Object.values(result).every(validUsageCount)) return left;
  const input: NonNullable<UsageMetadata["input_token_details"]> = {};
  const output: NonNullable<UsageMetadata["output_token_details"]> = {};
  for (const field of ["cache_read", "cache_creation"] as const) {
    const a = left.input_token_details?.[field], b = right.input_token_details?.[field];
    if (a !== undefined && b !== undefined && validUsageCount(a + b)) input[field] = a + b;
  }
  const a = left.output_token_details?.reasoning, b = right.output_token_details?.reasoning;
  if (a !== undefined && b !== undefined && validUsageCount(a + b)) output.reasoning = a + b;
  if (Object.keys(input).length) result.input_token_details = input;
  if (Object.keys(output).length) result.output_token_details = output;
  return result;
}

class RecoveryPayloadRouter {
  exposed = false;
  refusal?: string;
  finishReason?: string;
  private leading: ChatCompletionChunk[] = [];
  private readonly pending: ChatCompletionChunk[] = [];

  readonly content: string[] = [];
  readonly reasoning: string[] = [];
  readonly unsupportedRecoveryFields: Record<string, true> = {};

  constructor(private readonly rememberText: boolean) {}

  accept(payload: ChatCompletionChunk): ChatCompletionChunk[] {
    let choice = payload.error == null ? getStreamChoice(payload.choices) : undefined;
    const delta = choice?.delta ?? {};
    if (choice && this.recordCompletion(choice.finish_reason, delta.refusal)) {
      const { finish_reason: _finish, ...rest } = choice;
      choice = rest;
      payload = { ...payload, choices: [choice] };
    }
    this.captureContext(delta);
    const hasTools = Array.isArray(delta.tool_calls) ? delta.tool_calls.length > 0 : delta.tool_calls != null;
    if (!hasStreamOutput(delta)) {
      if (hasTools || this.pending.length || payload.usage != null) this.pending.push(payload);
      else if (this.exposed || payload.error != null) return [payload];
      else this.leading.push(payload);
      return [];
    }
    if (choice && (hasTools || payload.usage != null || (this.pending.length && choice.finish_reason != null))) {
      const [early, tool] = splitRecoveryPayload(payload, choice, delta);
      this.pending.push(tool);
      payload = early;
    }
    this.exposed = true;
    return [...this.leading.splice(0), payload];
  }

  finish(): ChatCompletionChunk[] { return [...this.leading, ...this.pending]; }

  private captureContext(delta: Record<string, unknown>): void {
    for (const field of ["audio", "annotations", "function_call"]) {
      if (delta[field] != null) this.unsupportedRecoveryFields[field] = true;
    }
    if (!this.rememberText) return;
    if (typeof delta.content === "string" && delta.content) this.content.push(delta.content);
    const reasoning = responseReasoning(delta, "stream delta");
    if (reasoning) this.reasoning.push(reasoning);
  }

  /** Record the finish reason and report repetitions before the SDK merges chunks. */
  private recordCompletion(finish: string | null | undefined, refusal: unknown): boolean {
    if (typeof refusal === "string" && refusal) this.refusal = refusal;
    if (finish == null) return false;
    if (this.finishReason === undefined) {
      this.finishReason = finish;
      return false;
    }
    if (this.finishReason !== finish) throw new Error("EXAONE returned conflicting stream finish reasons.");
    return true;
  }
}

function hasStreamOutput(delta: Record<string, unknown>): boolean {
  return ["content", "reasoning_content", "reasoning", "refusal", "function_call"].some((key) => Boolean(delta[key]));
}

function splitRecoveryPayload(
  payload: ChatCompletionChunk, choice: NonNullable<ChatCompletionChunk["choices"]>[number], delta: Record<string, unknown>,
): [ChatCompletionChunk, ChatCompletionChunk] {
  const { tool_calls, ...earlyDelta } = delta;
  const { usage: _usage, ...earlyPayload } = payload;
  const { finish_reason: _finish, matched_stop: _stop, ...earlyChoice } = choice;
  const { logprobs: _logprobs, ...heldChoice } = choice;
  return [
    { ...earlyPayload, choices: [{ ...earlyChoice, delta: earlyDelta }] },
    { ...payload, choices: [{ ...heldChoice, delta: tool_calls === undefined ? {} : { tool_calls } }] },
  ];
}

const correctionAccounting = Symbol("EXAONE tool call correction accounting");
type RecoveryAccounting = {
  usage: UsageMetadata | undefined;
  metadata: Record<string, unknown>;
  wireUsage: Record<string, unknown> | undefined;
};
type RecoveryPayload = ChatCompletionChunk & { [correctionAccounting]?: RecoveryAccounting };

interface RecoveryStreamState {
  message: Record<string, unknown>;
  calls: Map<number, { id: string; type: string; function: { name: string; arguments: string } }>;
  usage?: unknown;
  multipleIdFragments?: boolean;
}

function accumulateRecoveryPayload(state: RecoveryStreamState, payload: ChatCompletionChunk): void {
  if (payload.usage != null) state.usage = payload.usage;
  if (!Array.isArray(payload.choices) || payload.choices.length !== 1) return;
  const choice = payload.choices[0];
  if (!choice || !isRecord(choice.delta)) return;
  const delta = choice.delta;
  if (Array.isArray(delta.tool_calls)) delta.tool_calls.forEach((fragment) => accumulateRecoveryCall(state, fragment));
}

function accumulateRecoveryCall(state: RecoveryStreamState, fragment: unknown): void {
  if (!isRecord(fragment) || typeof fragment.index !== "number" || !Number.isSafeInteger(fragment.index) || fragment.index < 0) {
    throw new Error("EXAONE streamed tool call is missing a valid index.");
  }
  const call = state.calls.get(fragment.index) ?? { id: "", type: "function", function: { name: "", arguments: "" } };
  if (typeof fragment.id === "string") {
    if (call.id && fragment.id) state.multipleIdFragments = true;
    call.id += fragment.id;
  }
  if (typeof fragment.type === "string") call.type = fragment.type;
  if (isRecord(fragment.function)) {
    if (typeof fragment.function.name === "string") call.function.name += fragment.function.name;
    if (typeof fragment.function.arguments === "string") call.function.arguments += fragment.function.arguments;
  }
  state.calls.set(fragment.index, call);
}

function createAssistantMessage(
  source: Record<string, unknown>,
  response: ChatCompletionResponse,
  choice: NonNullable<ChatCompletionResponse["choices"]>[number],
  headers: Headers
): AIMessage<ExaoneMessageStructure> {
  validateAssistantTextFields(source);
  const refusal = typeof source.refusal === "string" ? source.refusal : undefined;
  const content = typeof source.content === "string" ? source.content : "";
  const additional_kwargs: Record<string, unknown> = {};
  const reasoning = responseReasoning(source, "assistant");
  if (reasoning !== undefined) additional_kwargs.reasoning_content = reasoning;
  if (refusal !== undefined) additional_kwargs.refusal = refusal;
  if (source.function_call !== undefined) additional_kwargs.function_call = source.function_call;
  for (const field of ["annotations", "audio"] as const) {
    if (source[field] != null) additional_kwargs[field] = source[field];
  }

  const rawToolCalls = toolCallEntries(source.tool_calls, "response");
  const [tool_calls, invalid_tool_calls] = parseAssistantToolCalls(rawToolCalls);
  const metadata = responseMetadata(response, choice, headers);
  if (rawToolCalls.length) metadata.tool_calls = rawToolCalls;
  const message = new AIMessage<ExaoneMessageStructure>({
    content,
    additional_kwargs,
    tool_calls,
    ...(invalid_tool_calls.length === 0 ? {} : { invalid_tool_calls }),
    response_metadata: metadata,
  });
  return message;
}

function parseAssistantToolCalls(rawCalls: Record<string, unknown>[]): ReturnType<typeof defaultToolCallParser> {
  const valid: ReturnType<typeof defaultToolCallParser>[0] = [];
  const invalid: ReturnType<typeof defaultToolCallParser>[1] = [];
  for (const raw of rawCalls) {
    const [parsed, rejected] = defaultToolCallParser([raw]);
    if (rejected.length) {
      const fn = raw.function as { arguments: string };
      const detail = invalidJsonDetail(fn.arguments);
      invalid.push(...rejected.map((call) => ({ ...call, error: detail })));
      continue;
    }
    const fn = raw.function as { name: string; arguments: string };
    if (!isRecord(JSON.parse(fn.arguments))) {
      invalid.push({ name: fn.name, args: fn.arguments, ...(typeof raw.id === "string" ? { id: raw.id } : {}), error: "Arguments must be a JSON object." });
    } else {
      valid.push(...parsed);
    }
  }
  return [valid, invalid];
}

function invalidJsonDetail(args: string): string {
  try {
    JSON.parse(args);
  } catch (error) {
    // Runtime error messages may include argument text; retain only the numeric position.
    if (error instanceof SyntaxError && error.message === "Unexpected end of JSON input") {
      return `Unexpected end of JSON at UTF-16 offset ${args.length}.`;
    }
    const match = error instanceof SyntaxError ? /\bposition (\d+)\b/.exec(error.message) : null;
    if (match) return `Invalid JSON at UTF-16 offset ${match[1]}.`;
  }
  return "Arguments were not valid JSON.";
}

function responseMetadata(
  response: ChatCompletionResponse | ChatCompletionChunk,
  choice: ChatCompletionChoiceMetadata | undefined,
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

function numericRecoveryUsage(value: unknown, secret: string): Record<string, unknown> | undefined {
  if (!isRecord(value)) return undefined;
  const result: Record<string, unknown> = {};
  for (const key of ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]) {
    if (validUsageCount(value[key])) result[key] = value[key];
  }
  for (const key of ["prompt_tokens_details", "completion_tokens_details"]) {
    const details = value[key];
    if (!isRecord(details)) continue;
    const safe = Object.fromEntries(Object.entries(details).filter(
      ([name, count]) => redactSecretText(name, secret) === name && validUsageCount(count),
    ));
    if (Object.keys(safe).length) result[key] = safe;
  }
  return Object.keys(result).length ? result : undefined;
}

class CorrectionFailureUsage {
  private correction?: Record<string, unknown>;

  constructor(private readonly failed: unknown, private readonly secret: string) {}

  record(payload: unknown): void {
    if (!isRecord(payload)) return;
    const usage = numericRecoveryUsage(payload.usage, this.secret);
    if (usage) this.correction = usage;
  }

  attach(error: unknown, signal: AbortSignal | undefined): void {
    // An abort reason belongs to the caller and can be reused across requests.
    if (signal?.aborted && error === signal.reason) return;
    if (!(error instanceof Error) || !Object.isExtensible(error)) return;
    const failedUsage = numericRecoveryUsage(this.failed, this.secret);
    const correctionUsage = this.correction;
    if (!failedUsage && !correctionUsage) return;
    try {
      Object.defineProperty(error, "exaone_recovery_usage", {
        value: { ...(failedUsage && { failed_usage: failedUsage }), ...(correctionUsage && { correction_usage: correctionUsage }) },
        configurable: true,
      });
    } catch {
      // Preserve the original error when its properties cannot be changed.
    }
  }
}

function validUsageCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function inputUsageDetails(value: unknown): UsageMetadata["input_token_details"] {
  if (!isRecord(value)) return undefined;
  const result: NonNullable<UsageMetadata["input_token_details"]> = {};
  if (validUsageCount(value.cached_tokens)) result.cache_read = value.cached_tokens;
  if (validUsageCount(value.cache_write_tokens)) result.cache_creation = value.cache_write_tokens;
  return Object.keys(result).length ? result : undefined;
}

function outputUsageDetails(usage: Record<string, unknown>): UsageMetadata["output_token_details"] {
  const details = isRecord(usage.completion_tokens_details) ? usage.completion_tokens_details : {};
  const reasoning = details.reasoning_tokens ?? usage.reasoning_tokens;
  const result: NonNullable<UsageMetadata["output_token_details"]> = {};
  if (validUsageCount(reasoning)) result.reasoning = reasoning;
  return Object.keys(result).length ? result : undefined;
}

function mapUsage(usage: Record<string, unknown> | undefined): UsageMetadata | undefined {
  if (!usage) return undefined;
  const input = usage.prompt_tokens;
  const output = usage.completion_tokens;
  const total = usage.total_tokens;
  if (!validUsageCount(input) || !validUsageCount(output) || !validUsageCount(total)) {
    return undefined;
  }
  const result: UsageMetadata = {
    input_tokens: input,
    output_tokens: output,
    total_tokens: total,
  };
  const inputDetails = inputUsageDetails(usage.prompt_tokens_details);
  const outputDetails = outputUsageDetails(usage);
  if (inputDetails) result.input_token_details = inputDetails;
  if (outputDetails) result.output_token_details = outputDetails;
  return result;
}

async function readResponseText(response: Response, apiKey: string): Promise<string> {
  try {
    return await response.text();
  } catch (error) {
    throw safeTransportError(error, apiKey);
  }
}

async function readJsonResponse<T>(response: Response, apiKey: string): Promise<T> {
  const text = await readResponseText(response, apiKey);
  if (!text) throw new Error("EXAONE returned an empty JSON response.");
  return parseResponseJson<T>(text);
}

function parseResponseJson<T>(text: string): T {
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error("EXAONE returned malformed JSON.");
  }
}

async function readErrorBody(response: Response, apiKey: string): Promise<unknown> {
  const text = await readResponseText(response, apiKey);
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { description: text };
  }
}

async function* parseSse<T>(
  body: ReadableStream<Uint8Array>, signal: AbortSignal | undefined, apiKey: string, requireDone = false,
  accounting?: CorrectionFailureUsage,
): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamEnded = false;
  let pendingCR = false;
  const cancelReader = () => {
    void reader.cancel().catch(() => undefined);
  };
  signal?.addEventListener("abort", cancelReader, { once: true });
  try {
    while (true) {
      signal?.throwIfAborted();
      const { done, value } = await readWithSignal(reader, signal);
      const text = done ? decoder.decode() : decoder.decode(value, { stream: true });
      streamEnded = done;
      const normalized = normalizeSseChunk(text, pendingCR);
      pendingCR = normalized.pendingCR;
      buffer += normalized.text;
      const frames = takeSseFrames(buffer);
      buffer = frames.remaining;
      const parsed = parseSseFrames<T>(frames.complete, accounting);
      yield* parsed.events;
      if (parsed.done) return;
      if (streamEnded) {
        yield* finalSseFrame<T>(buffer, requireDone);
        return;
      }
    }
  } catch (error) {
    if (signal?.aborted) throw signal.reason;
    throw safeTransportError(error, apiKey);
  } finally {
    signal?.removeEventListener("abort", cancelReader);
    if (!streamEnded) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

function finalSseFrame<T>(buffer: string, requireDone: boolean): T[] {
  const payload = ssePayload(buffer);
  if (payload === "[DONE]") return [];
  if (requireDone) throw new Error("EXAONE recovery stream ended before [DONE].");
  return payload ? [parseResponseJson<T>(payload)] : [];
}

function normalizeSseChunk(text: string, pendingCR: boolean): { text: string; pendingCR: boolean } {
  if (!text.length) return { text, pendingCR };
  // A CR ends a line immediately; ignore an LF that follows it in the next chunk.
  if (pendingCR && text.startsWith("\n")) text = text.slice(1);
  return { text: text.replace(/\r\n/g, "\n").replace(/\r/g, "\n"), pendingCR: text.endsWith("\r") };
}

function parseSseFrames<T>(frames: string[], accounting?: CorrectionFailureUsage): { events: T[]; done: boolean } {
  const events: T[] = [];
  for (const frame of frames) {
    const payload = ssePayload(frame);
    if (payload === "[DONE]") return { events, done: true };
    if (payload) {
      const event = parseResponseJson<T>(payload);
      accounting?.record(event);
      events.push(event);
    }
  }
  return { events, done: false };
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
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new TypeError("EXAONE baseUrl must be an absolute HTTP or HTTPS URL.");
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
