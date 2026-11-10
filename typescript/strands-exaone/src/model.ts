import { Model, ModelError, ModelNetworkError, ModelThrottledError, ContextWindowOverflowError } from '@strands-agents/sdk'
import type { Message, ModelStreamEvent, StreamOptions, StopReason } from '@strands-agents/sdk'
import { z } from 'zod'
import { formatMessages } from './messages.js'
import { redactMetadata, ResponseMetadataCollector } from './metadata.js'
import { sendWithRetry, validateRetryConfig } from './retry.js'
import { readSSE } from './sse.js'
import type { ExaoneMetadata, ExaoneModelConfig } from './types.js'

type WireCall = { index?: number; id?: string; function?: { name?: string; arguments?: string } }
type WireMessage = { content?: string | null; reasoning_content?: string | null; tool_calls?: WireCall[] | null }
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
  constructor(status: number, body: unknown, requestId: string | undefined, headers: Record<string, string>) {
    super(`EXAONE request failed with HTTP ${status}`)
    this.name = 'ExaoneAPIError'
    this.status = status
    this.body = body
    this.requestId = requestId
    this.headers = headers
  }
}

const rateHeaderNames = ['retry-after', 'retry-after-ms', 'x-second-rate-limit', 'x-second-rate-limit-remaining', 'x-minute-rate-limit', 'x-minute-rate-limit-remaining', 'x-day-rate-limit', 'x-day-rate-limit-remaining']
const MANAGED_PARAMS: ReadonlySet<string> = new Set(['model', 'messages', 'stream', 'stream_options', 'tools', 'tool_choice'])

function rateHeaders(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {}
  for (const name of rateHeaderNames) {
    const value = headers.get(name)
    if (value !== null) result[name] = value
  }
  return result
}

async function throwHttpError(response: Response, key: string): Promise<never> {
  const raw = (await response.text()).replaceAll(key, '[REDACTED]')
  let body: unknown = raw
  try { body = redactMetadata(JSON.parse(raw) as unknown, key) } catch { /* preserve non-JSON error */ }
  const requestId = response.headers.get('x-request-id')?.replaceAll(key, '[REDACTED]')
  const headers = redactMetadata(rateHeaders(response.headers), key) as Record<string, string>
  const error = new ExaoneAPIError(response.status, body, requestId, headers)
  if (response.status === 429) throw new ModelThrottledError(error.message, { cause: error })
  if (response.status === 400 && /context.{0,40}(length|window|limit|exceed)|maximum context/i.test(raw)) {
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
  if (template !== undefined && (typeof template !== 'object' || template === null || Array.isArray(template))) throw new TypeError('chat_template_kwargs must be an object')
  const templateOptions = { ...(template as Record<string, unknown> | undefined) }
  if (config.preserveThinking !== undefined) templateOptions.preserve_thinking = config.preserveThinking
  if (config.enableThinking !== undefined) templateOptions.enable_thinking = config.enableThinking
  if (Object.keys(templateOptions).length) params.chat_template_kwargs = templateOptions
  if (config.maxTokens !== undefined) params.max_tokens = config.maxTokens
  if (config.temperature !== undefined) params.temperature = config.temperature
  if (config.topP !== undefined) params.top_p = config.topP
  if (config.topK !== undefined) params.top_k = config.topK
  if (config.presencePenalty !== undefined) params.presence_penalty = config.presencePenalty
  if (config.reasoningEffort !== undefined) params.reasoning_effort = config.reasoningEffort
  return params
}

function requestParameters(config: ExaoneModelConfig, messages: Message[], options?: StreamOptions): Record<string, unknown> {
  const params = configuredParams(config)
  const stream = config.stream ?? true
  const result: Record<string, unknown> = { ...params, model: config.modelId ?? 'k-exaone_v2.5', messages: formatMessages(messages, options), stream }
  if (stream && config.includeUsage !== undefined) result.stream_options = { include_usage: config.includeUsage }
  if (options?.toolSpecs?.length) {
    result.tools = options.toolSpecs.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.inputSchema ?? { type: 'object', properties: {} } } }))
    if (options.toolChoice) result.tool_choice = makeToolChoice(options.toolChoice)
  }
  if (options?.structuredOutputSchema && !options.toolChoice && result.response_format === undefined) {
    result.response_format = { type: 'json_schema', json_schema: { name: 'structured_output', strict: true, schema: z.toJSONSchema(options.structuredOutputSchema) } }
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

/** Convert OpenAI-compatible deltas to non-overlapping Strands content blocks. */
class EventMapper {
  private active: TextKind | undefined
  private readonly calls = new Map<number, AccumulatedCall>()
  private finished: string | undefined
  usage: Record<string, unknown> | undefined
  readonly responseMetadata = new ResponseMetadataCollector()

  private textEvents(kind: TextKind, value: string): ModelStreamEvent[] {
    if (!value) return []
    const result: ModelStreamEvent[] = []
    if (kind !== this.active) {
      if (this.active) result.push({ type: 'modelContentBlockStopEvent' })
      result.push({ type: 'modelContentBlockStartEvent' })
      this.active = kind
    }
    result.push({ type: 'modelContentBlockDeltaEvent', delta: kind === 'reasoning' ? { type: 'reasoningContentDelta', text: value } : { type: 'textDelta', text: value } })
    return result
  }

  private updateCalls(calls: WireCall[]): void {
    for (const [position, call] of calls.entries()) {
      const index = call.index ?? position
      const current = this.calls.get(index) ?? { id: '', name: '', arguments: '' }
      if (call.id && call.id !== current.id) current.id += call.id
      if (call.function?.name && call.function.name !== current.name) current.name += call.function.name
      current.arguments += call.function?.arguments ?? ''
      this.calls.set(index, current)
    }
  }

  accept(event: WireResponse): ModelStreamEvent[] {
    if (event.error) throw new ModelError('EXAONE returned an error in the response stream')
    this.responseMetadata.accept(event)
    if (event.usage) this.usage = event.usage
    if ((event.choices?.length ?? 0) > 1) throw new ModelError('EXAONE returned multiple choices, which Strands cannot represent')
    const result: ModelStreamEvent[] = []
    for (const choice of event.choices ?? []) {
      if ((choice.index ?? 0) !== 0) throw new ModelError('EXAONE returned multiple choices, which Strands cannot represent')
      const delta = choice.delta ?? choice.message ?? {}
      result.push(...this.textEvents('reasoning', delta.reasoning_content ?? ''))
      result.push(...this.textEvents('text', delta.content ?? ''))
      if (delta.tool_calls) this.updateCalls(delta.tool_calls)
      if (choice.finish_reason) this.finished = choice.finish_reason
    }
    return result
  }

  finish(key: string): ModelStreamEvent[] {
    if (!this.finished) throw new ModelError('EXAONE stream ended without a finish reason')
    const result: ModelStreamEvent[] = []
    if (this.active) result.push({ type: 'modelContentBlockStopEvent' })
    const ids = new Set<string>()
    for (const [, call] of [...this.calls].sort(([a], [b]) => a - b)) {
      if (!call.id || !call.name) throw new ModelError('EXAONE returned an incomplete tool call')
      if (ids.has(call.id)) throw new ModelError('EXAONE returned duplicate tool call IDs')
      ids.add(call.id)
      let args: unknown
      try { args = JSON.parse(call.arguments) as unknown } catch { throw new ModelError('EXAONE returned invalid tool call arguments') }
      if (!args || typeof args !== 'object' || Array.isArray(args)) throw new ModelError('EXAONE tool call arguments must be an object')
      result.push({ type: 'modelContentBlockStartEvent', start: { type: 'toolUseStart', name: call.name, toolUseId: call.id } })
      result.push({ type: 'modelContentBlockDeltaEvent', delta: { type: 'toolUseInputDelta', input: call.arguments } })
      result.push({ type: 'modelContentBlockStopEvent' })
    }
    if (this.finished === 'tool_calls' && !this.calls.size) throw new ModelError('EXAONE returned tool_calls without a tool call')
    if (this.calls.size && this.finished !== 'tool_calls') throw new ModelError('EXAONE returned tool calls without a tool_calls finish reason')
    result.push({ type: 'modelMessageStopEvent', stopReason: finishReason(this.finished, key) })
    return result
  }
}

function metadataEvent(mapper: EventMapper, response: Response, key: string): { event: ModelStreamEvent; metadata: ExaoneMetadata } {
  const usage = mapper.usage
  const metadata: ExaoneMetadata = { rateLimit: rateHeaders(response.headers) }
  const requestId = response.headers.get('x-request-id')
  if (requestId !== null) metadata.requestId = requestId
  if (usage) metadata.usage = usage
  metadata.response = mapper.responseMetadata.snapshot()
  const safeMetadata = redactMetadata(metadata, key) as ExaoneMetadata
  const prompt = usage?.prompt_tokens
  const completion = usage?.completion_tokens
  const total = usage?.total_tokens
  const event: ModelStreamEvent = { type: 'modelMetadataEvent', trace: { exaone: safeMetadata } }
  if (typeof prompt === 'number' && typeof completion === 'number' && typeof total === 'number') {
    event.usage = { inputTokens: prompt, outputTokens: completion, totalTokens: total }
  }
  return { event, metadata: safeMetadata }
}

async function* responseEvents(response: Response, streaming: boolean, mapper: EventMapper): AsyncGenerator<ModelStreamEvent> {
  if (!streaming) {
    let data: WireResponse
    try { data = await response.json() as WireResponse } catch (error) {
      if (error instanceof SyntaxError) throw new ModelError('EXAONE returned malformed response JSON')
      throw error
    }
    yield* mapper.accept(data)
    return
  }
  if (!response.body) throw new ModelError('EXAONE returned an empty stream')
  let done = false
  for await (const payload of readSSE(response.body)) {
    if (payload === '[DONE]') { done = true; break }
    let data: WireResponse
    try { data = JSON.parse(payload) as WireResponse } catch { throw new ModelError('EXAONE returned malformed stream JSON') }
    yield* mapper.accept(data)
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

async function fetchResponse(config: ExaoneModelConfig, request: Record<string, unknown>, key: string, signal: AbortSignal): Promise<Response> {
  const transport = config.fetch ?? globalThis.fetch
  const url = `${(config.baseURL ?? 'https://api.exaone.ai/v1').replace(/\/+$/, '')}/chat/completions`
  try {
    return await transport(url, { method: 'POST', headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal })
  } catch (error) {
    if (signal.aborted) throw signal.reason
    throw new ModelNetworkError('EXAONE network request failed', { cause: safeTransportCause(error, key) })
  }
}

/** Strands model provider for EXAONE API Platform. */
export class ExaoneModel extends Model<ExaoneModelConfig> {
  private _config: ExaoneModelConfig
  constructor(config: ExaoneModelConfig = {}) {
    super()
    validateRetryConfig(config)
    this._config = { modelId: 'k-exaone_v2.5', contextWindowLimit: 524288, ...config }
  }
  override updateConfig(config: Partial<ExaoneModelConfig>): void {
    const updated = { ...this._config, ...config }
    validateRetryConfig(updated)
    this._config = updated
  }
  override getConfig(): ExaoneModelConfig {
    const { apiKey: _key, ...safe } = this._config
    return { ...safe }
  }
  override async *stream(messages: Message[], options?: StreamOptions): AsyncGenerator<ModelStreamEvent> {
    const config = { ...this._config }
    const key = getKey(config)
    const request = requestParameters(config, messages, options)
    const controller = new AbortController()
    const signals = [controller.signal]
    if (options?.cancelSignal) signals.push(options.cancelSignal)
    if (config.timeoutMs !== undefined) signals.push(AbortSignal.timeout(config.timeoutMs))
    const signal = AbortSignal.any(signals)
    const response = await sendWithRetry(config, signal, () => fetchResponse(config, request, key, signal))
    try {
      if (!response.ok) await throwHttpError(response, key)
      const mapper = new EventMapper()
      yield { type: 'modelMessageStartEvent', role: 'assistant' }
      try {
        yield* responseEvents(response, request.stream !== false, mapper)
      } catch (error) {
        if (signal.aborted) throw signal.reason
        if (error instanceof TypeError) throw new ModelNetworkError('EXAONE response stream failed', { cause: safeTransportCause(error, key) })
        throw error
      }
      for (const event of mapper.finish(key)) yield event
      const metadata = metadataEvent(mapper, response, key)
      config.onMetadata?.(metadata.metadata)
      yield metadata.event
    } finally { controller.abort() }
  }
}
