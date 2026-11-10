import { describe, expect, it, vi } from 'vitest'
import { Agent, Message, ModelError, ModelNetworkError, ModelThrottledError, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneAPIError, ExaoneModel } from '../src/index.js'
import { readSSE } from '../src/sse.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const usage = { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8, reasoning_tokens: 2 }
const wire = (choices: unknown[], extra: object = {}) => ({ choices, ...extra })
const event = (delta: unknown, finish: string | null = null) => wire([{ index: 0, delta, finish_reason: finish }])
const sse = (events: unknown[]) => events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n'
const response = (body: string, status = 200, headers: Record<string, string> = {}) => new Response(body, { status, headers })
const transport = (r: Response) => vi.fn(async () => r) as unknown as typeof fetch
async function aggregate(model: ExaoneModel, messages: Message[] = [user()]) {
  const iterator = model.streamAggregated(messages)
  while (true) {
    const item = await iterator.next()
    if (item.done) return item.value
  }
}
async function collect(model: ExaoneModel, messages: Message[] = [user()]) {
  const events = []
  for await (const item of model.stream(messages)) events.push(item)
  return events
}

describe('ExaoneModel', () => {
  it('preserves reasoning, text and usage while formatting request options', async () => {
    const data = sse([event({ reasoning_content: '생각' }), event({ content: ' 답' }), event({}, 'stop'), wire([], { usage })])
    const fetcher = vi.fn(async () => response(data, 200, { 'x-request-id': 'request-1' }))
    const onMetadata = vi.fn()
    const model = new ExaoneModel({ apiKey: 'private-key', fetch: fetcher as typeof fetch, reasoningEffort: 'medium', preserveThinking: true, enableThinking: false, topK: 0, maxTokens: 0, includeUsage: false, onMetadata })
    const result = await aggregate(model)
    expect(result.message.content.map((b) => b.type)).toEqual(['reasoningBlock', 'textBlock'])
    expect(result.message.content[0]).toMatchObject({ text: '생각' })
    expect(result.message.content[1]).toMatchObject({ text: ' 답' })
    expect(result.metadata?.usage).toEqual({ inputTokens: 3, outputTokens: 5, totalTokens: 8 })
    expect(result.metadata?.trace).toMatchObject({ exaone: { requestId: 'request-1', usage: { reasoning_tokens: 2 } } })
    expect(onMetadata).toHaveBeenCalledOnce()
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body).toMatchObject({ model: 'k-exaone_v2.5', max_tokens: 0, top_k: 0, reasoning_effort: 'medium', stream_options: { include_usage: false }, chat_template_kwargs: { preserve_thinking: true, enable_thinking: false } })
    expect(JSON.stringify(model.getConfig())).not.toContain('private-key')
  })

  it('executes tools and returns reasoning and tool results in the next agent turn', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return response(requests.length === 1
        ? sse([event({ reasoning_content: 'need lookup' }), event({ tool_calls: [{ index: 0, id: 'call-1', function: { name: 'lookup', arguments: '{}' } }] }), event({}, 'tool_calls')])
        : sse([event({ content: 'The number is 7.' }), event({}, 'stop')]))
    })
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up the number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher as typeof fetch, preserveThinking: true })
    const agent = new Agent({ model, tools: [lookup], printer: false })
    const result = await agent.invoke('Look up the number')
    expect(String(result)).toContain('7')
    expect(callback).toHaveBeenCalledOnce()
    const followUp = requests[1]!.messages as Record<string, unknown>[]
    expect(followUp).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'need lookup' }))
    expect(followUp).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'call-1', content: '7' }))
  })

  it('propagates cancellation and wraps response transport failures', async () => {
    const controller = new AbortController()
    const fetcher = ((_url: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), { once: true })
    })) as typeof fetch
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher })
    const pending = model.stream([user()], { cancelSignal: controller.signal }).next()
    controller.abort(new Error('cancelled by caller'))
    await expect(pending).rejects.toThrow('cancelled by caller')

    const stream = new ReadableStream<Uint8Array>({ start(streamController) { streamController.error(new TypeError('connection reset')) } })
    const broken = new ExaoneModel({ apiKey: 'key', fetch: transport(new Response(stream)) })
    await expect(collect(broken)).rejects.toBeInstanceOf(ModelNetworkError)
  })

  it('round trips assistant reasoning and sends tool results before the next user text', async () => {
    const fetcher = vi.fn(async () => response(JSON.stringify(wire([{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }], { usage }))))
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher as typeof fetch, stream: false })
    const messages = [
      Message.fromMessageData({ role: 'assistant', content: [{ reasoning: { text: 'reason' } }, { toolUse: { name: 'tool', toolUseId: 'id1', input: { x: 2 } } }] }),
      Message.fromMessageData({ role: 'user', content: [{ toolResult: { toolUseId: 'id1', status: 'success', content: [{ text: '7' }] } }, { text: 'next' }] }),
    ]
    await collect(model, messages)
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body.messages).toEqual([
      { role: 'assistant', content: null, reasoning_content: 'reason', tool_calls: [{ id: 'id1', type: 'function', function: { name: 'tool', arguments: '{"x":2}' } }] },
      { role: 'tool', tool_call_id: 'id1', content: '7' }, { role: 'user', content: 'next' },
    ])
  })

  it('serializes interleaved streamed tool fragments into two complete SDK blocks', async () => {
    const data = sse([
      event({ reasoning_content: 'r' }),
      event({ tool_calls: [{ index: 0, id: 'id-', function: { name: 'get_', arguments: '{"a":' } }, { index: 1, id: 'id2', function: { name: 'second', arguments: '{"b":' } }] }),
      event({ tool_calls: [{ index: 1, function: { arguments: '2}' } }, { index: 0, id: '1', function: { name: 'first', arguments: '1}' } }] }),
      event({}, 'tool_calls'), wire([], { usage }),
    ])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await aggregate(model)
    expect(result.stopReason).toBe('toolUse')
    expect(result.message.content.map((b) => b.type)).toEqual(['reasoningBlock', 'toolUseBlock', 'toolUseBlock'])
    expect(result.message.content[1]).toMatchObject({ toolUseId: 'id-1', name: 'get_first', input: { a: 1 } })
    expect(result.message.content[2]).toMatchObject({ toolUseId: 'id2', name: 'second', input: { b: 2 } })
  })

  it('keeps multiple non-streaming tool calls even though they lack index fields', async () => {
    const body = JSON.stringify(wire([{ index: 0, message: { tool_calls: [
      { id: 'a', function: { name: 'one', arguments: '{}' } },
      { id: 'b', function: { name: 'two', arguments: '{"n":2}' } },
    ] }, finish_reason: 'tool_calls' }], { usage }))
    const model = new ExaoneModel({ apiKey: 'key', stream: false, fetch: transport(response(body)) })
    const result = await aggregate(model)
    expect(result.message.content).toHaveLength(2)
    expect(result.message.content[0]).toMatchObject({ toolUseId: 'a', name: 'one', input: {} })
    expect(result.message.content[1]).toMatchObject({ toolUseId: 'b', name: 'two', input: { n: 2 } })
  })

  it('rejects missing and duplicate tool identifiers before returning any tool events', async () => {
    for (const calls of [
      [{ id: '', function: { name: 'fn', arguments: '{}' } }],
      [{ id: 'same', function: { name: 'one', arguments: '{}' } }, { id: 'same', function: { name: 'two', arguments: '{}' } }],
    ]) {
      const body = JSON.stringify(wire([{ message: { tool_calls: calls }, finish_reason: 'tool_calls' }]))
      const model = new ExaoneModel({ apiKey: 'key', stream: false, fetch: transport(response(body)) })
      const events = []
      try { for await (const item of model.stream([user()])) events.push(item) } catch (error) { expect(error).toBeInstanceOf(ModelError) }
      expect(events.some((item) => item.type === 'modelContentBlockStartEvent')).toBe(false)
    }
  })

  it.each(['', '{', 'null', '42', '[]'])('rejects incomplete or non-object tool arguments %s', async (args) => {
    const data = sse([event({ tool_calls: [{ index: 0, id: 'id', function: { name: 'fn', arguments: args } }] }), event({}, 'tool_calls')])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    await expect(collect(model)).rejects.toThrow(ModelError)
  })

  it('rejects truncated streams, missing finish reason, and multiple choices', async () => {
    const truncated = new ExaoneModel({ apiKey: 'key', fetch: transport(response('data: {"choices":[]}\n\n')) })
    await expect(collect(truncated)).rejects.toThrow('before [DONE]')
    const noFinish = new ExaoneModel({ apiKey: 'key', fetch: transport(response(sse([event({ content: 'partial' })]))) })
    await expect(collect(noFinish)).rejects.toThrow('without a finish reason')
    const multiple = new ExaoneModel({ apiKey: 'key', stream: false, fetch: transport(response(JSON.stringify(wire([{ index: 0 }, { index: 1 }]))) ) })
    await expect(collect(multiple)).rejects.toThrow('multiple choices')
  })

  it('maps rate limits and preserves the safe original error and retry headers', async () => {
    const body = JSON.stringify({ error: { message: 'private-key', type: 'rate_limit_error', code: 'rate_limit_exceeded' } })
    const model = new ExaoneModel({ apiKey: 'private-key', fetch: transport(response(body, 429, { 'retry-after-ms': '631', 'x-request-id': 'r1' })) })
    try { await collect(model); throw new Error('Expected rate limit') } catch (error) {
      expect(error).toBeInstanceOf(ModelThrottledError)
      const cause = (error as Error).cause as ExaoneAPIError
      expect(cause).toBeInstanceOf(ExaoneAPIError)
      expect(cause.status).toBe(429)
      expect(cause.headers['retry-after-ms']).toBe('631')
      expect(JSON.stringify(cause.body)).not.toContain('private-key')
    }
  })

  it('preserves nonstandard JSON errors and does not classify generic 400 as overflow', async () => {
    const detail = { detail: [{ msg: 'validation failed' }] }
    const validation = new ExaoneModel({ apiKey: 'key', fetch: transport(response(JSON.stringify(detail), 422)) })
    await expect(collect(validation)).rejects.toMatchObject({ status: 422, body: detail })
    const badRequest = new ExaoneModel({ apiKey: 'key', fetch: transport(response('{"error":{"message":"invalid request to model"}}', 400)) })
    await expect(collect(badRequest)).rejects.toBeInstanceOf(ExaoneAPIError)
  })

  it('formats structured output without overriding forced tool calls', async () => {
    const fetcher = vi.fn(async () => response(JSON.stringify(wire([{ index: 0, message: { content: '{"x":1}' }, finish_reason: 'stop' }]))))
    const model = new ExaoneModel({ apiKey: 'key', stream: false, fetch: fetcher as typeof fetch })
    for await (const _ of model.stream([user()], { structuredOutputSchema: z.object({ x: z.number() }) })) { /* consume */ }
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body.response_format).toMatchObject({ type: 'json_schema', json_schema: { strict: true } })
  })

  it('rejects multiple requested choices and unsupported image input', async () => {
    const model = new ExaoneModel({ apiKey: 'key', params: { n: 2 } })
    await expect(collect(model)).rejects.toThrow('n must be 1')
  })
})

describe('SSE parser', () => {
  it('handles CR-only, split CRLF, empty data fields and multi-line data', async () => {
    const encoder = new TextEncoder()
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      for (const part of ['data: one\r', '\ndata: two\r\n\r', '\ndata\r\rdata: ', 'last\r\r']) controller.enqueue(encoder.encode(part))
      controller.close()
    } })
    const events = []
    for await (const value of readSSE(stream)) events.push(value)
    expect(events).toEqual(['one\ntwo', '', 'last'])
  })
  it('preserves split UTF-8 bytes', async () => {
    const bytes = new TextEncoder().encode('data: 안녕\n\n')
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
      controller.close()
    } })
    const events = []
    for await (const value of readSSE(stream)) events.push(value)
    expect(events).toEqual(['안녕'])
  })
})
