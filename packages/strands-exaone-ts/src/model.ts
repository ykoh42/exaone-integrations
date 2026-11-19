/**
 * EXAONE Chat Completions model provider implementation.
 *
 * @see https://docs.exaone.ai/en/docs/k-exaone-2.5
 */
import { Model, ModelError, ModelNetworkError, ModelThrottledError, ContextWindowOverflowError } from '@strands-agents/sdk'
import type { Message, ModelInfo, ModelStreamEvent, StreamOptions, StopReason } from '@strands-agents/sdk'
import { z } from 'zod'
import { copyConfig, validateModelConfig } from './config.js'
import { formatMessages } from './messages.js'
import { redactMetadata, ResponseMetadataCollector } from './metadata.js'
import { sendWithRetry } from './retry.js'
import { readSSE, ResponseReadError } from './sse.js'
import type { ExaoneMetadata, ExaoneModelConfig } from './types.js'
import { InvalidToolArgumentsError, recoveryJsonDiagnostic } from './recovery-error.js'

interface WireCallFunction {
  name?: string | null
  arguments?: string | null
}
interface WireCall {
  index?: number
  id?: string | null
  type?: string | null
  function?: WireCallFunction | null
}
interface WireMessage {
  content?: string | null
  reasoning_content?: string | null
  reasoning?: string | null
  refusal?: string | null
  tool_calls?: WireCall[] | null
}
interface WireChoice {
  index?: number
  delta?: WireMessage
  message?: WireMessage
  finish_reason?: string | null
}
type WireResponse = { choices?: WireChoice[]; usage?: Record<string, unknown> | null; error?: unknown } & Record<string, unknown>
interface AccumulatedCall {
  id: string
  idFragments: number
  name: string
  arguments: string
}
type TextKind = 'reasoning' | 'text'

function recoverableTools(request: Record<string, unknown>): ReadonlySet<string> {
  const tools = Array.isArray(request.tools) ? request.tools : []
  const names = new Set<string>()
  for (const tool of tools) {
    if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') names.add(tool.function.name)
  }
  const choice = request.tool_choice
  const named = isRecord(choice) && choice.type === 'function' && isRecord(choice.function) ? choice.function.name : undefined
  if (choice == null || choice === 'auto' || choice === 'required') return names
  return new Set(typeof named === 'string' && names.has(named) ? [named] : [])
}

/** An HTTP error with EXAONE's response body and selected diagnostic headers. */
export class ExaoneAPIError extends ModelError {
  readonly status: number
  readonly body: unknown
  readonly requestId: string | undefined
  readonly headers: Record<string, string>
  readonly errorType: string | undefined
  readonly code: unknown
  readonly param: unknown
  constructor(status: number, body: unknown, requestId: string | undefined, headers: Record<string, string>) {
    super(errorMessage(body, `HTTP ${status}`))
    this.name = 'ExaoneAPIError'
    this.status = status
    this.body = body
    this.requestId = requestId
    this.headers = headers
    const details = errorDetails(body)
    this.errorType = details.errorType
    this.code = details.code
    this.param = details.param
  }
}

/** An error returned in a successful HTTP response body or event stream. */
export class ExaoneResponseError extends ModelError {
  readonly body: unknown
  readonly errorType: string | undefined
  readonly code: unknown
  readonly param: unknown

  constructor(body: unknown) {
    super(errorMessage(body, 'EXAONE returned an error in the response body'))
    this.name = 'ExaoneResponseError'
    this.body = body
    const details = errorDetails(body)
    this.errorType = details.errorType
    this.code = details.code
    this.param = details.param
  }
}

function errorMessage(body: unknown, fallback: string): string {
  const envelope = isRecord(body) ? body : {}
  const nested = isRecord(envelope.error) ? envelope.error : {}
  const message = nested.message || envelope.description
  return message === undefined || message === null || message === '' ? fallback : String(message)
}

function errorDetails(body: unknown): { errorType: string | undefined; code: unknown; param: unknown } {
  const envelope = isRecord(body) ? body : {}
  const nested = isRecord(envelope.error) ? envelope.error : {}
  return {
    errorType: typeof nested.type === 'string' ? nested.type : undefined,
    code: 'code' in nested ? nested.code : envelope.result_code,
    param: nested.param,
  }
}

const RATE_LIMIT_HEADERS = [
  'retry-after', 'retry-after-ms',
  'x-second-rate-limit', 'x-second-rate-limit-remaining',
  'x-minute-rate-limit', 'x-minute-rate-limit-remaining',
  'x-day-rate-limit', 'x-day-rate-limit-remaining',
]
const MANAGED_PARAMS: ReadonlySet<string> = new Set(['model', 'messages', 'stream', 'tools', 'tool_choice'])

function rateHeaders(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {}
  for (const name of RATE_LIMIT_HEADERS) {
    const value = headers.get(name)
    if (value !== null) result[name] = value
  }
  return result
}

function requestTimeout(ms: number | undefined): { signal: AbortSignal; dispose: () => void } | undefined {
  if (ms === undefined) return undefined
  if (typeof ms !== 'number') throw new TypeError('timeoutMs must be a number')
  if (!Number.isInteger(ms) || ms < 0 || ms > 4_294_967_295) throw new RangeError('timeoutMs must be an integer between 0 and 4294967295')
  const controller = new AbortController()
  const maxTimeout = 2_147_483_647
  let remaining = ms
  let timer: ReturnType<typeof setTimeout>
  const schedule = () => {
    const delay = Math.min(remaining, maxTimeout)
    remaining -= delay
    timer = setTimeout(() => {
      if (remaining > 0) schedule()
      else controller.abort(new DOMException('The operation was aborted due to timeout', 'TimeoutError'))
    }, delay)
  }
  schedule()
  return { signal: controller.signal, dispose: () => clearTimeout(timer) }
}

async function throwHttpError(response: Response, key: string, signal: AbortSignal): Promise<never> {
  let body: unknown
  let readCause: Error | undefined
  try {
    const raw = redactMetadata(await response.text(), key) as string
    body = raw
    try { body = redactMetadata(JSON.parse(raw) as unknown, key) } catch { /* preserve non-JSON error */ }
  } catch (error) {
    if (signal.aborted) throw signal.reason
    readCause = safeResponseReadCause(error)
  }
  const rawRequestId = response.headers.get('x-request-id')
  const requestId = rawRequestId === null ? undefined : redactMetadata(rawRequestId, key) as string
  const headers = redactMetadata(rateHeaders(response.headers), key) as Record<string, string>
  const error = new ExaoneAPIError(response.status, body, requestId, headers)
  if (readCause) Object.defineProperty(error, 'cause', { value: readCause })
  if (response.status === 429) throw new ModelThrottledError(error.message, { cause: error })
  if ((response.status === 400 || response.status === 413) && isRecord(body) && isRecord(body.error) && body.error.code === 'context_length_exceeded') {
    const overflow = new ContextWindowOverflowError('EXAONE context window exceeded')
    Object.defineProperty(overflow, 'cause', { value: error })
    throw overflow
  }
  throw error
}

function getKey(config: ExaoneModelConfig): string {
  const environment = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
  const key = config.apiKey ?? environment?.EXAONE_API_KEY
  if (!key) {
    throw new Error("EXAONE API key is required. Provide it via the 'apiKey' option or set the EXAONE_API_KEY environment variable.")
  }
  return key
}

function makeToolChoice(choice: NonNullable<StreamOptions['toolChoice']>): unknown {
  if ('auto' in choice) return 'auto'
  if ('any' in choice) return 'required'
  return { type: 'function', function: { name: choice.tool.name } }
}

function checkManagedParams(params: Record<string, unknown>): void {
  for (const key of MANAGED_PARAMS) if (key in params) throw new TypeError(`Parameter ${key} is managed by the provider`)
}

function configuredParams(config: ExaoneModelConfig): Record<string, unknown> {
  const params = { ...config.params }
  checkManagedParams(params)
  if (params.n !== undefined && params.n !== 1) throw new TypeError('Strands supports one choice; n must be 1')
  const template = params.chat_template_kwargs
  if (template !== undefined && (typeof template !== 'object' || template === null || Array.isArray(template))) {
    throw new TypeError('chat_template_kwargs must be an object')
  }
  const templateOptions = { ...(template as Record<string, unknown> | undefined) }
  if (config.preserveThinking !== undefined) templateOptions.preserve_thinking = config.preserveThinking
  else if (templateOptions.preserve_thinking === undefined) templateOptions.preserve_thinking = true
  if (config.enableThinking !== undefined) templateOptions.enable_thinking = config.enableThinking
  params.chat_template_kwargs = templateOptions
  if (config.maxTokens !== undefined) params.max_tokens = config.maxTokens
  if (config.temperature !== undefined) params.temperature = config.temperature
  else if (params.temperature === undefined) params.temperature = 1.0
  if (config.topP !== undefined) params.top_p = config.topP
  else if (params.top_p === undefined) params.top_p = 0.95
  if (config.topK !== undefined) params.top_k = config.topK
  if (config.presencePenalty !== undefined) params.presence_penalty = config.presencePenalty
  if (config.reasoningEffort !== undefined) params.reasoning_effort = config.reasoningEffort
  return params
}

function requestParameters(config: ExaoneModelConfig, messages: Message[], options?: StreamOptions): Record<string, unknown> {
  const params = configuredParams(config)
  const stream = config.stream ?? true
  const result: Record<string, unknown> = {
    ...params,
    model: config.modelId ?? 'k-exaone_v2.5',
    messages: formatMessages(messages, options),
    stream,
  }
  if (stream) {
    const streamOptions = params.stream_options
    if (streamOptions !== undefined && streamOptions !== null && (typeof streamOptions !== 'object' || Array.isArray(streamOptions))) {
      throw new TypeError('stream_options must be an object or null')
    }
    const resolvedStreamOptions = { ...(streamOptions as Record<string, unknown> | null | undefined) }
    if (config.includeUsage !== undefined) resolvedStreamOptions.include_usage = config.includeUsage
    else if (!('include_usage' in resolvedStreamOptions)) resolvedStreamOptions.include_usage = true
    result.stream_options = resolvedStreamOptions
  } else {
    delete result.stream_options
  }
  if (options?.toolSpecs?.length) {
    result.tools = options.toolSpecs.map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema ?? { type: 'object', properties: {} },
      },
    }))
    if (options.toolChoice) result.tool_choice = makeToolChoice(options.toolChoice)
  }
  if (options?.structuredOutputSchema && !options.toolChoice && result.response_format === undefined) {
    result.response_format = {
      type: 'json_schema',
      json_schema: { name: 'structured_output', strict: true, schema: z.toJSONSchema(options.structuredOutputSchema) },
    }
  }
  return result
}

function finishReason(reason: string, key: string): StopReason {
  switch (reason) {
    case 'stop': return 'endTurn'
    case 'tool_calls': return 'toolUse'
    case 'length': return 'maxTokens'
    case 'content_filter': return 'contentFiltered'
    default: {
      const safeReason = (redactMetadata(String(reason), key) as string).slice(0, 128)
      throw new ModelError(`Unrecognized EXAONE finish reason: ${JSON.stringify(safeReason)}`)
    }
  }
}

function toolFragment(value: unknown, field: string): string {
  if (value === null || value === undefined) return ''
  if (typeof value !== 'string') throw new ModelError(`EXAONE returned an invalid tool ${field}; expected a string`)
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function validateWireMessage(message: unknown): void {
  if (message == null) return
  if (!isRecord(message)) throw new ModelError('EXAONE returned an invalid message')
  const calls = message.tool_calls
  if (calls == null) return
  if (!Array.isArray(calls)) throw new ModelError('EXAONE returned invalid tool calls')
  for (const call of calls) {
    if (!isRecord(call) || (call.function != null && !isRecord(call.function))) {
      throw new ModelError('EXAONE returned an invalid tool call')
    }
  }
}

/** Validate containers before accessing their fields; contents are validated by the mapper. */
function wireResponse(value: unknown): WireResponse {
  if (!isRecord(value)) throw new ModelError('EXAONE returned an invalid response object')
  if (value.error != null || value.choices == null) return value as WireResponse
  if (!Array.isArray(value.choices)) throw new ModelError('EXAONE returned invalid choices')
  for (const choice of value.choices) {
    if (!isRecord(choice)) throw new ModelError('EXAONE returned an invalid choice')
    validateWireMessage(choice.delta)
    validateWireMessage(choice.message)
  }
  return value as WireResponse
}

/** Normalize the two compatible server fields without dropping conflicting data. */
function reasoningText(message: WireMessage): string | undefined {
  const content = message.reasoning_content
  const reasoning = message.reasoning
  if ((content != null && typeof content !== 'string') || (reasoning != null && typeof reasoning !== 'string')) {
    throw new ModelError('EXAONE returned invalid reasoning content; expected a string')
  }
  if (content && reasoning && content !== reasoning) {
    throw new ModelError('EXAONE returned conflicting reasoning and reasoning_content values')
  }
  return content || reasoning || undefined
}

/** Convert OpenAI-compatible deltas to non-overlapping Strands content blocks. */
class EventMapper {
  private active: TextKind | undefined
  private readonly calls = new Map<number, AccumulatedCall>()
  private finished: string | undefined
  private refusal = ''
  private hasNonWhitespaceContent = false
  usage: Record<string, unknown> | undefined
  readonly responseMetadata = new ResponseMetadataCollector()

  constructor(private readonly key: string, private readonly namedToolChoice: boolean,
    private readonly recoveryTools?: ReadonlySet<string>, private readonly recoveryToolName?: string) {}

  private textEvents(kind: TextKind, value: unknown): ModelStreamEvent[] {
    if (value === null || value === undefined || value === '') return []
    if (typeof value !== 'string') throw new ModelError(`EXAONE returned invalid ${kind} content; expected a string`)
    const result: ModelStreamEvent[] = []
    if (kind !== this.active) {
      if (this.active) result.push({ type: 'modelContentBlockStopEvent' })
      result.push({ type: 'modelContentBlockStartEvent' })
      this.active = kind
    }
    result.push({
      type: 'modelContentBlockDeltaEvent',
      delta: kind === 'reasoning' ? { type: 'reasoningContentDelta', text: value } : { type: 'textDelta', text: value },
    })
    return result
  }

  private updateCalls(calls: WireCall[], streaming: boolean): void {
    for (const [position, call] of calls.entries()) {
      if (call.type != null && call.type !== 'function') throw new ModelError('EXAONE returned an unsupported tool call type')
      const index = call.index === undefined && !streaming ? position : call.index
      if (typeof index !== 'number' || !Number.isSafeInteger(index) || index < 0) {
        throw new ModelError('EXAONE returned a tool call without a valid index')
      }
      const current = this.calls.get(index) ?? { id: '', idFragments: 0, name: '', arguments: '' }
      const id = toolFragment(call.id, 'ID')
      if (id) current.idFragments += 1
      current.id += id
      current.name += toolFragment(call.function?.name, 'name')
      current.arguments += toolFragment(call.function?.arguments, 'arguments')
      this.calls.set(index, current)
    }
  }

  accept(event: WireResponse, streaming: boolean): ModelStreamEvent[] {
    this.validateResponse(event)
    if (this.recoveryToolName === undefined) this.responseMetadata.accept(event)
    if (isRecord(event.usage)) this.usage = event.usage
    const result: ModelStreamEvent[] = []
    for (const choice of event.choices ?? []) result.push(...this.acceptChoice(choice, streaming))
    return result
  }

  private validateResponse(event: WireResponse): void {
    if (event.error != null) throw new ExaoneResponseError(redactMetadata(event, this.key))
    if ((event.choices?.length ?? 0) > 1) throw new ModelError('EXAONE returned multiple choices, which Strands cannot represent')
  }

  private readonly recoveryText: string[] = []
  private readonly recoveryReasoning: string[] = []

  recoveryMessages(error: InvalidToolArgumentsError): Record<string, unknown>[] {
    const assistant: Record<string, unknown> = {
      role: 'assistant', content: this.recoveryText.join(''),
      tool_calls: [{ id: error.callId, type: 'function', function: { name: error.toolName, arguments: '{}' } }],
    }
    if (this.recoveryReasoning.length) assistant.reasoning_content = this.recoveryReasoning.join('')
    const diagnostic = JSON.stringify({ raw_arguments: error.rawArguments, parse_error: error.diagnostic })
    return [assistant, {
      role: 'tool', tool_call_id: error.callId,
      content: '[ERROR] Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n'
        + diagnostic + '\nCall ' + error.toolName + ' once with a valid JSON object.',
    }]
  }

  replaceToolArguments(correction: EventMapper, key: string): void {
    if (correction.finished === 'length') {
      this.calls.clear()
      this.finished = 'length'
      return
    }
    correction.finish(key)
    this.calls.values().next().value!.arguments = correction.calls.values().next().value!.arguments
  }

  private acceptChoice(choice: WireChoice, streaming: boolean): ModelStreamEvent[] {
    if ((choice.index ?? 0) !== 0) throw new ModelError('EXAONE returned multiple choices, which Strands cannot represent')
    const delta = choice.delta ?? choice.message ?? {}
    const reasoning = reasoningText(delta)
    if (this.recoveryTools?.size) {
      if (reasoning) this.recoveryReasoning.push(reasoning)
      if (delta.content) this.recoveryText.push(delta.content)
    }
    const result = this.textEvents('reasoning', reasoning)
    this.acceptRefusal(delta.refusal)
    if (typeof delta.content === 'string' && delta.content.trim().length > 0) this.hasNonWhitespaceContent = true
    result.push(...this.textEvents('text', delta.content))
    if (delta.tool_calls) this.updateCalls(delta.tool_calls, streaming)
    this.acceptFinishReason(choice.finish_reason)
    return result
  }

  private acceptRefusal(refusal: unknown): void {
    if (refusal !== null && refusal !== undefined && typeof refusal !== 'string') {
      throw new ModelError('EXAONE returned invalid refusal content; expected a string')
    }
    if (refusal) this.refusal += refusal
  }

  private acceptFinishReason(reason: string | null | undefined): void {
    if (!reason) return
    if (this.finished && this.finished !== reason) throw new ModelError('EXAONE returned conflicting finish reasons')
    this.finished = reason
  }

  finish(key: string): ModelStreamEvent[] {
    if (!this.finished) throw new ModelError('EXAONE stream ended without a finish reason')
    const namedToolStop = this.namedToolChoice && this.calls.size > 0 && this.finished === 'stop'
    if (this.recoveryToolName !== undefined && this.finished !== 'length') this.validateReplacement(namedToolStop)
    const result: ModelStreamEvent[] = []
    this.appendRefusal(result)
    if (this.active) result.push({ type: 'modelContentBlockStopEvent' })
    this.appendToolCalls(result)
    this.validateToolFinishReason()
    result.push({ type: 'modelMessageStopEvent', stopReason: namedToolStop ? 'toolUse' : finishReason(this.finished, key) })
    return result
  }

  private validateReplacement(namedToolStop: boolean): void {
    if (this.refusal || (this.finished !== 'tool_calls' && !namedToolStop) || this.calls.size !== 1) {
      throw new ModelError('EXAONE recovery must return exactly one completed tool call')
    }
    const call = this.calls.values().next().value!
    if (call.idFragments > 1) throw new ModelError('EXAONE recovery returned an ambiguous streamed tool call ID')
    if (call.name !== this.recoveryToolName || !call.arguments.trim()) {
      throw new ModelError('EXAONE recovery must return the original tool with a JSON object')
    }
  }

  private appendRefusal(events: ModelStreamEvent[]): void {
    if (!this.refusal) return
    if (this.calls.size && (this.recoveryTools !== undefined || !this.hasNonWhitespaceContent)) {
      throw new ModelError('EXAONE returned refusal text together with tool calls')
    }
    if (!this.hasNonWhitespaceContent) events.push(...this.textEvents('text', this.refusal))
  }

  private appendToolCalls(events: ModelStreamEvent[]): void {
    const ids = new Set<string>()
    for (const [, call] of [...this.calls].sort(([a], [b]) => a - b)) this.appendToolCall(call, ids, events)
  }

  private appendToolCall(call: AccumulatedCall, ids: Set<string>, events: ModelStreamEvent[]): void {
    if (!call.id || !call.name) throw new ModelError('EXAONE returned an incomplete tool call')
    if (ids.has(call.id)) throw new ModelError('EXAONE returned duplicate tool call IDs')
    ids.add(call.id)
    this.checkRecoverable(call)
    // The SDK gives maxTokens precedence over a partial tool-input parse error.
    // Forward the original fragment; do not complete it or infer executability.
    if (this.finished !== 'length') this.validateToolInput(call.arguments)
    events.push({ type: 'modelContentBlockStartEvent', start: { type: 'toolUseStart', name: call.name, toolUseId: call.id } })
    events.push({ type: 'modelContentBlockDeltaEvent', delta: { type: 'toolUseInputDelta', input: call.arguments } })
    events.push({ type: 'modelContentBlockStopEvent' })
  }

  private checkRecoverable(call: AccumulatedCall): void {
    if (!this.recoveryTools?.has(call.name) || this.calls.size !== 1 || this.finished === 'length' || this.refusal || !call.arguments.trim()) return
    try { JSON.parse(call.arguments) } catch (error) {
      if (call.idFragments > 1) throw new ModelError('Cannot recover a tool call with ambiguous streamed ID fragments')
      const values = [call.id, call.name, call.arguments]
      const containsKey = values.some((value) => redactMetadata(value, this.key) !== value)
      if (!(this.finished === 'tool_calls' || (this.namedToolChoice && this.finished === 'stop')) || containsKey) {
        throw new ModelError('EXAONE returned invalid tool call arguments')
      }
      const detail = recoveryJsonDiagnostic(error, call.arguments)
      const failure = new InvalidToolArgumentsError(call.id, call.name, call.arguments, detail)
      failure.usage = this.usage === undefined ? undefined : redactMetadata(this.usage, this.key) as Record<string, unknown>
      throw failure
    }
  }

  private validateToolInput(input: string): void {
    if (!input.trim()) return
    let args: unknown
    try { args = JSON.parse(input) as unknown } catch {
      throw new ModelError('EXAONE returned invalid tool call arguments')
    }
    if (!isRecord(args)) throw new ModelError('EXAONE tool call arguments must be an object')
  }

  private validateToolFinishReason(): void {
    if (this.finished === 'tool_calls' && !this.calls.size) throw new ModelError('EXAONE returned tool_calls without a tool call')
    if (this.calls.size && this.finished !== 'tool_calls' && this.finished !== 'length' && !(this.namedToolChoice && this.finished === 'stop')) {
      throw new ModelError('EXAONE returned tool calls without a tool_calls finish reason')
    }
  }
}

function validTokenCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

type MetadataEvent = Extract<ModelStreamEvent, { type: 'modelMetadataEvent' }>

function modelUsage(usage: Record<string, unknown> | undefined): MetadataEvent['usage'] {
  const prompt = usage?.prompt_tokens
  const completion = usage?.completion_tokens
  const total = usage?.total_tokens
  if (!validTokenCount(prompt) || !validTokenCount(completion) || !validTokenCount(total)) return undefined
  const result: NonNullable<MetadataEvent['usage']> = { inputTokens: prompt, outputTokens: completion, totalTokens: total }
  if (isRecord(usage?.prompt_tokens_details)) {
    const { cached_tokens: read, cache_write_tokens: write } = usage.prompt_tokens_details
    if (validTokenCount(read)) result.cacheReadInputTokens = read
    if (validTokenCount(write)) result.cacheWriteInputTokens = write
  }
  return result
}

function metadataEvent(mapper: EventMapper, response: Response, key: string): { event: MetadataEvent; metadata: ExaoneMetadata } {
  const usage = mapper.usage
  const metadata: ExaoneMetadata = { status: response.status, rateLimit: rateHeaders(response.headers) }
  const requestId = response.headers.get('x-request-id')
  if (requestId !== null) metadata.requestId = requestId
  if (usage) metadata.usage = usage
  metadata.response = mapper.responseMetadata.snapshot()
  const safeMetadata = redactMetadata(metadata, key) as ExaoneMetadata
  const event: MetadataEvent = { type: 'modelMetadataEvent', trace: { exaone: safeMetadata } }
  const counts = modelUsage(usage)
  if (counts) event.usage = counts
  return { event, metadata: safeMetadata }
}

/** Keep the original response metadata and include both requests in SDK usage. */
function addRecoveryUsage(event: MetadataEvent, metadata: ExaoneMetadata, failed: InvalidToolArgumentsError,
  correction: { usage: Record<string, unknown> | undefined }, key: string,
): void {
  const correctionUsage = redactMetadata(correction.usage, key) as Record<string, unknown> | undefined
  metadata.recovery = {
    failedAttempts: [failed.usage === undefined ? {} : { usage: failed.usage }],
    ...(correctionUsage === undefined ? {} : { correctionUsage }),
  }
  const original = event.usage
  const corrected = modelUsage(correction.usage)
  if (!corrected) return
  event.usage = corrected
  if (!original || !failed.usage) return
  const previous = failed.usage
  const counts = [previous.prompt_tokens, previous.completion_tokens, previous.total_tokens]
  if (!counts.every(validTokenCount)) return
  const [input, output, total] = counts as [number, number, number]
  const totals = [event.usage.inputTokens + input, event.usage.outputTokens + output, event.usage.totalTokens + total]
  if (!totals.every(validTokenCount)) return
  event.usage.inputTokens = totals[0] as number
  event.usage.outputTokens = totals[1] as number
  event.usage.totalTokens = totals[2] as number
  // Cache totals cannot be inferred when either response omits a count.
  const details = isRecord(previous.prompt_tokens_details) ? previous.prompt_tokens_details : {}
  for (const [key, source] of [['cacheReadInputTokens', 'cached_tokens'], ['cacheWriteInputTokens', 'cache_write_tokens']] as const) {
    const old = details[source], current = event.usage[key]
    if (validTokenCount(old) && validTokenCount(current) && validTokenCount(old + current)) event.usage[key] = old + current
    else delete event.usage[key]
  }
}

async function notifyMetadata(callback: ExaoneModelConfig['onMetadata'], metadata: ExaoneMetadata, signal: AbortSignal): Promise<void> {
  if (signal.aborted) throw signal.reason
  if (!callback) return
  let result: void | Promise<void>
  try {
    result = callback(metadata)
  } catch (error) {
    if (signal.aborted) throw signal.reason
    throw new ModelError('EXAONE metadata callback failed', { cause: error })
  }
  await new Promise<void>((resolve, reject) => {
    const abort = () => {
      signal.removeEventListener('abort', abort)
      reject(signal.reason)
    }
    signal.addEventListener('abort', abort, { once: true })
    Promise.resolve(result).then(
      () => { signal.removeEventListener('abort', abort); resolve() },
      (error: unknown) => { signal.removeEventListener('abort', abort); reject(new ModelError('EXAONE metadata callback failed', { cause: error })) }
    )
    if (signal.aborted) abort()
  })
}

async function* withMessageStart(
  source: AsyncIterable<ModelStreamEvent> | Iterable<ModelStreamEvent>,
  signal: AbortSignal,
  state: { started: boolean },
): AsyncGenerator<ModelStreamEvent> {
  for await (const event of source) {
    signal.throwIfAborted()
    if (!state.started) {
      state.started = true
      yield { type: 'modelMessageStartEvent', role: 'assistant' }
      signal.throwIfAborted()
    }
    yield event
    signal.throwIfAborted()
  }
}

async function readResponseJSON(response: Response): Promise<WireResponse> {
  let raw: string
  try { raw = await response.text() } catch (error) { throw new ResponseReadError(error) }
  let data: unknown
  try { data = JSON.parse(raw) as unknown } catch { throw new ModelError('EXAONE returned malformed response JSON') }
  return wireResponse(data)
}

async function* responseEvents(response: Response, streaming: boolean, mapper: EventMapper): AsyncGenerator<ModelStreamEvent> {
  if (!streaming) { yield* mapper.accept(await readResponseJSON(response), false); return }
  if (!response.body) throw new ModelError('EXAONE returned an empty stream')
  let done = false
  for await (const payload of readSSE(response.body)) {
    if (payload === '[DONE]') { done = true; break }
    let data: unknown
    try { data = JSON.parse(payload) as unknown } catch { throw new ModelError('EXAONE returned malformed stream JSON') }
    yield* mapper.accept(wireResponse(data), true)
  }
  if (!done) throw new ModelError('EXAONE stream ended before [DONE]')
}

/** Copy safe diagnostics without retaining an untrusted transport's nested cause or stack. */
function safeTransportCause(error: unknown, key: string): Error {
  if (!(error instanceof Error)) return new Error('Transport failed')
  const message = redactMetadata(error.message, key) as string
  const safe = error instanceof TypeError ? new TypeError(message) : new Error(message)
  safe.name = redactMetadata(error.name, key) as string
  return safe
}

/** Read failures may contain credentials or request content, so expose no original diagnostics. */
function safeResponseReadCause(error: unknown): Error {
  if (error instanceof SyntaxError) return new SyntaxError('EXAONE response body read failed')
  if (error instanceof TypeError) return new TypeError('EXAONE response body read failed')
  return new Error('EXAONE response body read failed')
}

/** Serialize once so retries send the same request, without exposing input on failure. */
function requestBody(request: Record<string, unknown>): string | undefined {
  try { return JSON.stringify(request) } catch { throw new ModelError('EXAONE request could not be serialized as JSON') }
}

async function fetchResponse(config: ExaoneModelConfig, body: string | undefined, key: string, signal: AbortSignal): Promise<Response> {
  const transport = config.fetch ?? globalThis.fetch
  const url = `${(config.baseURL ?? 'https://api.exaone.ai/v1').replace(/\/+$/, '')}/chat/completions`
  try {
    return await transport(url, { method: 'POST', headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body }), signal })
  } catch (error) {
    if (signal.aborted) throw signal.reason
    throw new ModelNetworkError('EXAONE network request failed', { cause: safeTransportCause(error, key) })
  }
}

function namedChoiceUnsupported(error: unknown): boolean {
  if (!(error instanceof ExaoneAPIError) || ![400, 422, 501].includes(error.status)) return false
  if (error.code === 'unsupported_tool_choice' || (error.param === 'tool_choice' && error.code === 'unsupported_value')) return true
  return /named (?:function|tool)|(?:function|object)[- ](?:type|valued) tool_choice/i.test(error.message) &&
    /not support|unsupported/i.test(error.message)
}

function correctionError(error: unknown): unknown {
  // The agent's transport retry may not replay the original invocation.
  if (error instanceof ModelNetworkError || error instanceof ModelThrottledError || error instanceof ContextWindowOverflowError) {
    return new ModelError('EXAONE tool-call correction failed', { cause: error })
  }
  return error
}

/**
 * EXAONE model provider for Strands Agents.
 *
 * The API key defaults to the `EXAONE_API_KEY` environment variable. Previous
 * reasoning is sent back with assistant messages. `preserveThinking` defaults
 * to `true` to request retention across user turns.
 *
 * @example
 * ```typescript
 * const model = new ExaoneModel({ preserveThinking: true })
 * const agent = new Agent({ model })
 * ```
 */
export class ExaoneModel extends Model<ExaoneModelConfig> {
  #config: ExaoneModelConfig
  #recovery = false
  constructor(config: ExaoneModelConfig = {}) {
    super()
    this.#config = { modelId: 'k-exaone_v2.5', ...copyConfig(config) }
    validateModelConfig(this.#config)
  }
  /** Update the model configuration. Invalid options leave the current configuration unchanged. */
  override updateConfig(config: Partial<ExaoneModelConfig>): void {
    const updated = { ...this.#config, ...copyConfig(config) }
    validateModelConfig(updated)
    this.#config = updated
  }
  /** Get the model configuration without the API key. */
  override getConfig(): ExaoneModelConfig {
    const { apiKey: _key, ...safe } = this.#config
    return copyConfig(safe)
  }
  /** Known EXAONE capabilities, with caller overrides taking precedence. */
  override get modelInfo(): ModelInfo {
    const info = super.modelInfo
    if (this.#config.modelId !== 'k-exaone_v2.5' || (this.#config.baseURL ?? 'https://api.exaone.ai/v1').replace(/\/+$/, '') !== 'https://api.exaone.ai/v1') return info
    const overrides = this.#config.modelInfo?.capabilities
    return Object.freeze({
      ...info,
      limits: Object.freeze({
        ...info.limits,
        contextWindow: this.#config.modelInfo?.limits?.contextWindow ?? this.#config.contextWindowLimit ?? 524_288,
      }),
      capabilities: Object.freeze({
        ...info.capabilities,
        stateful: overrides?.stateful ?? false,
        forcedToolChoice: overrides?.forcedToolChoice ?? true,
        nativeStructuredOutput: overrides?.nativeStructuredOutput ?? true,
      }),
    })
  }
  /** @internal Create an isolated model for recovery without changing the agent's model. */
  recoveryModel(): ExaoneModel {
    const model = new ExaoneModel(this.#config)
    model.#recovery = true
    return model
  }

  /** Stream an EXAONE response as Strands model events. */
  override async *stream(messages: Message[], options?: StreamOptions): AsyncGenerator<ModelStreamEvent> {
    const config = { ...this.#config }
    const key = getKey(config)
    const request = requestParameters(config, messages, options)
    const recoveryTools = this.#recovery ? recoverableTools(request) : undefined
    const body = requestBody(request)
    const controller = new AbortController()
    const timeout = requestTimeout(config.timeoutMs)
    try {
      const signals = [controller.signal, options?.cancelSignal, timeout?.signal].filter((item): item is AbortSignal => item != null)
      const signal = AbortSignal.any(signals)
      const response = await sendWithRetry(config, signal, () => fetchResponse(config, body, key, signal))
      signal.throwIfAborted()
      if (!response.ok) await throwHttpError(response, key, signal)
      const mapper = new EventMapper(key, !!options?.toolSpecs?.length && !!options.toolChoice && 'tool' in options.toolChoice,
        recoveryTools)
      const messageStartState = { started: false }
      try {
        yield* withMessageStart(responseEvents(response, request.stream !== false, mapper), signal, messageStartState)
      } catch (error) {
        if (signal.aborted) throw signal.reason
        if (error instanceof ResponseReadError) {
          const cause = safeResponseReadCause(error.cause)
          if (messageStartState.started) throw new ModelError('EXAONE response stream failed after output was exposed', { cause })
          throw new ModelNetworkError('EXAONE response stream failed', { cause })
        }
        throw error
      }
      const completed = await this.#finish(mapper, request, config, key, signal)
      yield* withMessageStart(completed.events, signal, messageStartState)
      const metadata = metadataEvent(mapper, response, key)
      if (completed.correction) {
        addRecoveryUsage(metadata.event, metadata.metadata, completed.correction.failure, completed.correction, key)
      }
      if (config.onMetadata) await notifyMetadata(config.onMetadata, structuredClone(metadata.metadata), signal)
      signal.throwIfAborted()
      yield metadata.event
    } finally { timeout?.dispose(); controller.abort() }
  }

  async #finish(mapper: EventMapper, request: Record<string, unknown>, config: ExaoneModelConfig, key: string, signal: AbortSignal) {
    try {
      return { events: mapper.finish(key) }
    } catch (error) {
      if (!(error instanceof InvalidToolArgumentsError)) throw error
      const usage = await this.#correct(mapper, error, request, config, key, signal)
      return { events: mapper.finish(key), correction: { usage, failure: error } }
    }
  }

  async #correct(original: EventMapper, error: InvalidToolArgumentsError, request: Record<string, unknown>,
    config: ExaoneModelConfig, key: string, signal: AbortSignal,
  ): Promise<Record<string, unknown> | undefined> {
    const corrected: Record<string, unknown> = {
      ...request, messages: [...(request.messages as Record<string, unknown>[]), ...original.recoveryMessages(error)],
      tool_choice: { type: 'function', function: { name: error.toolName } },
    }
    const send = async () => {
      signal.throwIfAborted()
      const body = requestBody(corrected)
      const response = await sendWithRetry(config, signal, () => fetchResponse(config, body, key, signal))
      if (!response.ok) await throwHttpError(response, key, signal)
      return response
    }
    let response: Response
    try { response = await send() } catch (rejection) {
      if (!namedChoiceUnsupported(rejection)) throw correctionError(rejection)
      corrected.tool_choice = 'required'
      try { response = await send() } catch (failure) { throw correctionError(failure) }
    }
    const mapper = new EventMapper(key, corrected.tool_choice !== 'required', new Set(), error.toolName)
    try {
      for await (const _ of responseEvents(response, request.stream !== false, mapper)) signal.throwIfAborted()
    } catch (failure) {
      if (signal.aborted) throw signal.reason
      if (failure instanceof ResponseReadError) {
        throw new ModelError('EXAONE correction response stream failed', { cause: safeResponseReadCause(failure.cause) })
      }
      throw failure
    }
    signal.throwIfAborted()
    original.replaceToolArguments(mapper, key)
    return mapper.usage
  }

}
