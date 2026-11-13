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

type WireCall = {
  index?: number
  id?: string | null
  type?: string | null
  function?: { name?: string | null; arguments?: string | null } | null
}
type WireMessage = { content?: string | null; reasoning_content?: string | null; refusal?: string | null; tool_calls?: WireCall[] | null }
type WireChoice = { index?: number; delta?: WireMessage; message?: WireMessage; finish_reason?: string | null }
type WireResponse = { choices?: WireChoice[]; usage?: Record<string, unknown> | null; error?: unknown } & Record<string, unknown>
type AccumulatedCall = { id: string; name: string; arguments: string }
type TextKind = 'reasoning' | 'text'

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
    const raw = (await response.text()).replaceAll(key, '[REDACTED]')
    body = raw
    try { body = redactMetadata(JSON.parse(raw) as unknown, key) } catch { /* preserve non-JSON error */ }
  } catch (error) {
    if (signal.aborted) throw signal.reason
    readCause = safeResponseReadCause(error)
  }
  const requestId = response.headers.get('x-request-id')?.replaceAll(key, '[REDACTED]')
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
    const options = { ...(streamOptions as Record<string, unknown> | null | undefined) }
    if (config.includeUsage !== undefined) options.include_usage = config.includeUsage
    else if (!('include_usage' in options)) options.include_usage = true
    result.stream_options = options
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
      const safeReason = String(reason).replaceAll(key, '[REDACTED]').slice(0, 128)
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

/** Mark structural characters while respecting JSON string escapes. */
function jsonOutsideStringMask(input: string): boolean[] | undefined {
  const outside: boolean[] = []
  let inString = false
  let escaped = false
  for (let index = 0; index < input.length; index += 1) {
    const char = input[index]!
    if (inString) {
      outside.push(false)
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') inString = false
    } else {
      outside.push(char !== '"')
      if (char === '"') inString = true
    }
  }
  return inString || escaped ? undefined : outside
}

/** Remove a comma only when the next token closes its JSON container. */
function removeTrailingCommas(input: string, outside: boolean[]): { value: string; changed: boolean } {
  const output: string[] = []
  let changed = false
  for (let index = 0; index < input.length; index += 1) {
    const char = input[index]!
    if (char === ',' && outside[index]) {
      let next = index + 1
      while (next < input.length && /\s/.test(input[next]!)) next += 1
      if (input[next] === '}' || input[next] === ']') {
        changed = true
        continue
      }
    }
    output.push(char)
  }
  return { value: output.join(''), changed }
}

/** Return missing object/array closers, or undefined for mismatched delimiters. */
function openJsonClosers(input: string, outside: boolean[]): string[] | undefined {
  const closers: string[] = []
  for (let index = 0; index < input.length; index += 1) {
    if (!outside[index]) continue
    const char = input[index]!
    if (char === '{' || char === '[') closers.push(char === '{' ? '}' : ']')
    else if ((char === '}' || char === ']') && closers.pop() !== char) return undefined
  }
  return closers
}

/** Complete only mechanically recoverable JSON syntax; never infer argument values. */
function closeToolArguments(input: string): string | undefined {
  let outside = jsonOutsideStringMask(input)
  if (outside === undefined) return undefined
  const commas = removeTrailingCommas(input, outside)
  let candidate = commas.value
  outside = jsonOutsideStringMask(candidate)
  if (outside === undefined) return undefined
  const closers = openJsonClosers(candidate, outside)
  if (closers === undefined) return undefined
  let changed = commas.changed
  if (closers.length) {
    const lastSignificant = candidate.trimEnd().at(-1)
    if (!lastSignificant || '{[:,'.includes(lastSignificant)) return undefined
    candidate += closers.reverse().join('')
    changed = true
  }
  if (!changed) return undefined

  try {
    const parsed: unknown = JSON.parse(candidate)
    return isRecord(parsed) ? candidate : undefined
  } catch {
    return undefined
  }
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

/** Convert OpenAI-compatible deltas to non-overlapping Strands content blocks. */
class EventMapper {
  private active: TextKind | undefined
  private readonly calls = new Map<number, AccumulatedCall>()
  private finished: string | undefined
  private refusal = ''
  private hasNonWhitespaceContent = false
  usage: Record<string, unknown> | undefined
  readonly responseMetadata = new ResponseMetadataCollector()

  constructor(private readonly key: string) {}

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
      const current = this.calls.get(index) ?? { id: '', name: '', arguments: '' }
      current.id += toolFragment(call.id, 'ID')
      current.name += toolFragment(call.function?.name, 'name')
      current.arguments += toolFragment(call.function?.arguments, 'arguments')
      this.calls.set(index, current)
    }
  }

  accept(event: WireResponse, streaming: boolean): ModelStreamEvent[] {
    this.validateResponse(event)
    this.responseMetadata.accept(event)
    if (isRecord(event.usage)) this.usage = event.usage
    const result: ModelStreamEvent[] = []
    for (const choice of event.choices ?? []) result.push(...this.acceptChoice(choice, streaming))
    return result
  }

  private validateResponse(event: WireResponse): void {
    if (event.error != null) throw new ExaoneResponseError(redactMetadata(event, this.key))
    if ((event.choices?.length ?? 0) > 1) throw new ModelError('EXAONE returned multiple choices, which Strands cannot represent')
  }

  private acceptChoice(choice: WireChoice, streaming: boolean): ModelStreamEvent[] {
    if ((choice.index ?? 0) !== 0) throw new ModelError('EXAONE returned multiple choices, which Strands cannot represent')
    const delta = choice.delta ?? choice.message ?? {}
    const result = this.textEvents('reasoning', delta.reasoning_content)
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
    const result: ModelStreamEvent[] = []
    this.appendRefusal(result)
    if (this.active) result.push({ type: 'modelContentBlockStopEvent' })
    this.appendToolCalls(result)
    this.validateToolFinishReason()
    result.push({ type: 'modelMessageStopEvent', stopReason: finishReason(this.finished, key) })
    return result
  }

  private appendRefusal(events: ModelStreamEvent[]): void {
    if (!this.refusal || this.hasNonWhitespaceContent) return
    if (this.calls.size) throw new ModelError('EXAONE returned refusal text together with tool calls')
    events.push(...this.textEvents('text', this.refusal))
  }

  private appendToolCalls(events: ModelStreamEvent[]): void {
    const ids = new Set<string>()
    for (const [, call] of [...this.calls].sort(([a], [b]) => a - b)) this.appendToolCall(call, ids, events)
  }

  private appendToolCall(call: AccumulatedCall, ids: Set<string>, events: ModelStreamEvent[]): void {
    if (!call.id || !call.name) throw new ModelError('EXAONE returned an incomplete tool call')
    if (ids.has(call.id)) throw new ModelError('EXAONE returned duplicate tool call IDs')
    ids.add(call.id)
    // Blank arguments represent a zero-argument tool; emit JSON the SDK can aggregate as an object.
    let input = call.arguments.trim() ? call.arguments : '{}'
    let args: unknown
    try { args = JSON.parse(input) as unknown } catch {
      const repaired = closeToolArguments(input)
      if (repaired === undefined) throw new ModelError('EXAONE returned invalid tool call arguments')
      input = repaired
      args = JSON.parse(input) as unknown
    }
    if (!args || typeof args !== 'object' || Array.isArray(args)) throw new ModelError('EXAONE tool call arguments must be an object')
    events.push({ type: 'modelContentBlockStartEvent', start: { type: 'toolUseStart', name: call.name, toolUseId: call.id } })
    events.push({ type: 'modelContentBlockDeltaEvent', delta: { type: 'toolUseInputDelta', input } })
    events.push({ type: 'modelContentBlockStopEvent' })
  }

  private validateToolFinishReason(): void {
    if (this.finished === 'tool_calls' && !this.calls.size) throw new ModelError('EXAONE returned tool_calls without a tool call')
    if (this.calls.size && this.finished !== 'tool_calls') throw new ModelError('EXAONE returned tool calls without a tool_calls finish reason')
  }
}

function validTokenCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function metadataEvent(mapper: EventMapper, response: Response, key: string): { event: ModelStreamEvent; metadata: ExaoneMetadata } {
  const usage = mapper.usage
  const metadata: ExaoneMetadata = { status: response.status, rateLimit: rateHeaders(response.headers) }
  const requestId = response.headers.get('x-request-id')
  if (requestId !== null) metadata.requestId = requestId
  if (usage) metadata.usage = usage
  metadata.response = mapper.responseMetadata.snapshot()
  const safeMetadata = redactMetadata(metadata, key) as ExaoneMetadata
  const prompt = usage?.prompt_tokens
  const completion = usage?.completion_tokens
  const total = usage?.total_tokens
  const event: ModelStreamEvent = { type: 'modelMetadataEvent', trace: { exaone: safeMetadata } }
  if (validTokenCount(prompt) && validTokenCount(completion) && validTokenCount(total)) {
    event.usage = { inputTokens: prompt, outputTokens: completion, totalTokens: total }
  }
  return { event, metadata: safeMetadata }
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
  const message = error.message.replaceAll(key, '[REDACTED]')
  const safe = error instanceof TypeError ? new TypeError(message) : new Error(message)
  safe.name = error.name.replaceAll(key, '[REDACTED]')
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
  private _config: ExaoneModelConfig
  constructor(config: ExaoneModelConfig = {}) {
    super()
    this._config = { modelId: 'k-exaone_v2.5', contextWindowLimit: 524288, ...copyConfig(config) }
    validateModelConfig(this._config)
  }
  /** Update the model configuration. Invalid options leave the current configuration unchanged. */
  override updateConfig(config: Partial<ExaoneModelConfig>): void {
    const updated = { ...this._config, ...copyConfig(config) }
    validateModelConfig(updated)
    this._config = updated
  }
  /** Get the model configuration without the API key. */
  override getConfig(): ExaoneModelConfig {
    const { apiKey: _key, ...safe } = this._config
    return copyConfig(safe)
  }
  /** Known EXAONE capabilities, with caller overrides taking precedence. */
  override get modelInfo(): ModelInfo {
    const info = super.modelInfo
    if (this._config.modelId !== 'k-exaone_v2.5') return info
    const overrides = this._config.modelInfo?.capabilities
    return Object.freeze({
      ...info,
      limits: Object.freeze({
        ...info.limits,
        contextWindow: this._config.modelInfo?.limits?.contextWindow ?? this._config.contextWindowLimit ?? 524_288,
      }),
      capabilities: Object.freeze({
        ...info.capabilities,
        stateful: overrides?.stateful ?? false,
        forcedToolChoice: overrides?.forcedToolChoice ?? true,
        nativeStructuredOutput: overrides?.nativeStructuredOutput ?? true,
      }),
    })
  }
  /**
   * Stream an EXAONE response as Strands model events.
   *
   * @param messages - Conversation history to send to the model.
   * @param options - System prompt, tools, structured output, and cancellation options.
   * @yields Strands model events translated from the response.
   */
  override async *stream(messages: Message[], options?: StreamOptions): AsyncGenerator<ModelStreamEvent> {
    const config = { ...this._config }
    const key = getKey(config)
    const request = requestParameters(config, messages, options)
    const body = requestBody(request)
    const controller = new AbortController()
    const timeout = requestTimeout(config.timeoutMs)
    try {
      const signals = [controller.signal, options?.cancelSignal, timeout?.signal].filter((item): item is AbortSignal => item != null)
      const signal = AbortSignal.any(signals)
      const response = await sendWithRetry(config, signal, () => fetchResponse(config, body, key, signal))
      signal.throwIfAborted()
      if (!response.ok) await throwHttpError(response, key, signal)
      const mapper = new EventMapper(key)
      yield { type: 'modelMessageStartEvent', role: 'assistant' }
      signal.throwIfAborted()
      let deliveredContent = false
      try {
        for await (const event of responseEvents(response, request.stream !== false, mapper)) {
          signal.throwIfAborted()
          if (event.type === 'modelContentBlockDeltaEvent') deliveredContent = true
          yield event
          signal.throwIfAborted()
        }
      } catch (error) {
        if (signal.aborted) throw signal.reason
        if (error instanceof ResponseReadError) {
          const cause = safeResponseReadCause(error.cause)
          if (deliveredContent) throw new ModelError('EXAONE response stream failed after content was delivered', { cause })
          throw new ModelNetworkError('EXAONE response stream failed', { cause })
        }
        throw error
      }
      for (const event of mapper.finish(key)) {
        signal.throwIfAborted()
        yield event
        signal.throwIfAborted()
      }
      const metadata = metadataEvent(mapper, response, key)
      await notifyMetadata(config.onMetadata, structuredClone(metadata.metadata), signal)
      signal.throwIfAborted()
      yield metadata.event
    } finally { timeout?.dispose(); controller.abort() }
  }
}
