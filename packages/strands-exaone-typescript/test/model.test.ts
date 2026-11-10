import { describe, expect, it, vi } from 'vitest'
import { Agent, Message, ModelError, ModelNetworkError, ModelThrottledError, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneAPIError, ExaoneModel, ExaoneResponseError } from '../src/index.js'
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
  it('explains both ways to provide the API key', async () => {
    await expect(collect(new ExaoneModel({ apiKey: '' }))).rejects.toThrow(/apiKey.*EXAONE_API_KEY/)
  })

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

  it.each([
    { id: 'a', name: 'b' },
    { id: 'call_1', name: 'lookup' },
  ])('concatenates repeated streamed ID and name fragments', async ({ id, name }) => {
    const data = sse([
      event({ tool_calls: [{ index: 0, id, function: { name, arguments: '{}' } }] }),
      event({ tool_calls: [{ index: 0, id, function: { name } }] }),
      event({}, 'tool_calls'),
    ])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await aggregate(model)
    expect(result.message.content[0]).toMatchObject({ toolUseId: id + id, name: name + name, input: {} })
  })

  it('assembles unambiguous streamed tool ID and name fragments', async () => {
    const data = sse([
      event({ tool_calls: [{ index: 0, id: 'a', function: { name: 'x', arguments: '{"n":' } }] }),
      event({ tool_calls: [{ index: 0, id: 'b', function: { name: 'y', arguments: '1}' } }] }),
      event({}, 'tool_calls'),
    ])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await aggregate(model)
    expect(result.message.content[0]).toMatchObject({ toolUseId: 'ab', name: 'xy', input: { n: 1 } })
  })

  it('keeps repeated argument deltas in order', async () => {
    const data = sse([
      event({ tool_calls: [{ index: 0, id: 'call', function: { name: 'lookup', arguments: '{"word":"' } }] }),
      event({ tool_calls: [{ index: 0, function: { arguments: 'a' } }] }),
      event({ tool_calls: [{ index: 0, function: { arguments: 'a' } }] }),
      event({ tool_calls: [{ index: 0, function: { arguments: '"}' } }] }),
      event({}, 'tool_calls'),
    ])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await aggregate(model)
    expect(result.message.content[0]).toMatchObject({ toolUseId: 'call', name: 'lookup', input: { word: 'aa' } })
  })

  it.each([
    {
      idParts: ['a', 'a'], nameParts: ['a', 'a'], argumentParts: ['{', '"v":', '1', '}'],
      expectedId: 'aa', expectedName: 'aa', expectedInput: { v: 1 },
    },
    {
      idParts: ['call_', null, 'id', ''], nameParts: ['get', undefined, '_', 'item'],
      argumentParts: ['{"city":"', '서', '울', '"}'],
      expectedId: 'call_id', expectedName: 'get_item', expectedInput: { city: '서울' },
    },
    {
      idParts: ['i', '', 'd'], nameParts: ['name', 'name'], argumentParts: ['{"word":"', 'a', 'a', '"}'],
      expectedId: 'id', expectedName: 'namename', expectedInput: { word: 'aa' },
    },
    {
      idParts: [null, 'call'], nameParts: ['', 'lookup'], argumentParts: [null, '{}'],
      expectedId: 'call', expectedName: 'lookup', expectedInput: {},
    },
  ])('reassembles valid deltas with varied boundaries and optional fragments', async (testCase) => {
    const count = Math.max(testCase.idParts.length, testCase.nameParts.length, testCase.argumentParts.length)
    const chunks = Array.from({ length: count }, (_, index) => event({
      tool_calls: [{
        index: 0,
        id: testCase.idParts[index],
        type: index === 0 ? 'function' : null,
        function: { name: testCase.nameParts[index], arguments: testCase.argumentParts[index] },
      }],
    }))
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(sse([...chunks, event({}, 'tool_calls')]))) })
    const result = await aggregate(model)
    expect(result.message.content[0]).toMatchObject({
      toolUseId: testCase.expectedId, name: testCase.expectedName, input: testCase.expectedInput,
    })
  })

  it('executes only the tool whose name matches the combined streamed fragments', async () => {
    const data = sse([
      event({ tool_calls: [{ index: 0, id: 'call_', type: 'function', function: { name: 'a', arguments: '{}' } }] }),
      event({ tool_calls: [{ index: 0, id: '1', function: { name: 'a' } }] }),
      event({}, 'tool_calls'),
    ])
    const shortCallback = vi.fn(() => 'incorrect')
    const fullCallback = vi.fn(() => 'correct')
    const shortTool = tool({ name: 'a', description: 'Short name', inputSchema: z.object({}), callback: shortCallback })
    const fullTool = tool({ name: 'aa', description: 'Full name', inputSchema: z.object({}), callback: fullCallback })
    let count = 0
    const fetcher = vi.fn(async () => response(count++ === 0 ? data : sse([event({ content: 'done' }), event({}, 'stop')])))
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher as typeof fetch })
    const agent = new Agent({ model, tools: [shortTool, fullTool], printer: false, retryStrategy: null })
    await agent.invoke('Use the tool')
    expect(shortCallback).not.toHaveBeenCalled()
    expect(fullCallback).toHaveBeenCalledOnce()
  })

  it('rejects an unsupported tool call type before exposing or executing a tool', async () => {
    const data = sse([
      event({ tool_calls: [{ index: 0, id: 'call_1', type: 'unexpected_type', function: { name: 'lookup', arguments: '{}' } }] }),
      event({}, 'tool_calls'),
    ])
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up a number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const agent = new Agent({ model, tools: [lookup], printer: false, retryStrategy: null })
    await expect(agent.invoke('Look up the number')).rejects.toThrow(/unsupported tool call type/i)
    expect(callback).not.toHaveBeenCalled()
    const events = []
    const second = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    await expect((async () => {
      for await (const item of second.stream([user()])) events.push(item)
    })()).rejects.toThrow(ModelError)
    expect(events.some((item) => item.type === 'modelContentBlockStartEvent' && item.start?.type === 'toolUseStart')).toBe(false)
  })

  it.each([undefined, null, -1, 0.5, '0'])('rejects invalid streaming tool index %s before tool events', async (index) => {
    const data = sse([
      event({ tool_calls: [{ index, id: 'call_1', function: { name: 'lookup', arguments: '{}' } }] }),
      event({}, 'tool_calls'),
    ])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const events = []
    await expect((async () => {
      for await (const item of model.stream([user()])) events.push(item)
    })()).rejects.toThrow(/valid index/i)
    expect(events.some((item) => item.type === 'modelContentBlockStartEvent' && item.start?.type === 'toolUseStart')).toBe(false)
  })

  it('does not execute a tool whose streaming index is missing', async () => {
    const data = sse([
      event({ tool_calls: [{ id: 'call_1', function: { name: 'lookup', arguments: '{}' } }] }),
      event({}, 'tool_calls'),
    ])
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up a number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const agent = new Agent({ model, tools: [lookup], printer: false, retryStrategy: null })
    await expect(agent.invoke('Look up the number')).rejects.toThrow(/valid index/i)
    expect(callback).not.toHaveBeenCalled()
  })

  it.each([
    { field: 'reasoning_content', stream: true },
    { field: 'content', stream: true },
    { field: 'reasoning_content', stream: false },
    { field: 'content', stream: false },
  ])('rejects non-string $field without executing tools (stream=$stream)', async ({ field, stream }) => {
    const call = { id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }
    const payload = stream
      ? sse([event({ [field]: { value: 'invalid' }, tool_calls: [{ index: 0, ...call }] }, 'tool_calls')])
      : JSON.stringify(wire([{ index: 0, message: { [field]: { value: 'invalid' }, tool_calls: [call] }, finish_reason: 'tool_calls' }]))
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up a number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', stream, fetch: transport(response(payload)) })
    const agent = new Agent({ model, tools: [lookup], printer: false, retryStrategy: null })
    await expect(agent.invoke('Look up the number')).rejects.toThrow(/expected a string/i)
    expect(callback).not.toHaveBeenCalled()
  })

  it.each([
    { field: 'id', stream: true },
    { field: 'name', stream: true },
    { field: 'arguments', stream: true },
    { field: 'id', stream: false },
    { field: 'name', stream: false },
    { field: 'arguments', stream: false },
  ])('rejects non-string tool $field without executing tools (stream=$stream)', async ({ field, stream }) => {
    const call = {
      index: 0,
      id: field === 'id' ? 42 : 'call_1',
      type: 'function',
      function: { name: field === 'name' ? 42 : 'lookup', arguments: field === 'arguments' ? 42 : '{}' },
    }
    const payload = stream
      ? sse([event({ tool_calls: [call] }, 'tool_calls')])
      : JSON.stringify(wire([{ index: 0, message: { tool_calls: [call] }, finish_reason: 'tool_calls' }]))
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up a number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', stream, fetch: transport(response(payload)) })
    const agent = new Agent({ model, tools: [lookup], printer: false, retryStrategy: null })
    await expect(agent.invoke('Look up the number')).rejects.toThrow(/expected a string/i)
    expect(callback).not.toHaveBeenCalled()
  })

  it('accepts null streamed tool fragments before the complete values arrive', async () => {
    const data = sse([
      event({ tool_calls: [{ index: 0, id: null, function: { name: null, arguments: null } }] }),
      event({ tool_calls: [{ index: 0, id: 'call_1', function: { name: 'lookup', arguments: '{}' } }] }),
      event({}, 'tool_calls'),
    ])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await aggregate(model)
    expect(result.message.content[0]).toMatchObject({ toolUseId: 'call_1', name: 'lookup', input: {} })
  })

  it('exposes redacted SSE error bodies and codes without describing HTTP 200 as an error', async () => {
    const body = { error: { message: 'private-key', code: 'code-private-key', type: 'model_error' }, 'private-key': 'secret' }
    const model = new ExaoneModel({ apiKey: 'private-key', fetch: transport(response(sse([body]))) })
    try {
      await collect(model)
      throw new Error('Expected a response error')
    } catch (error) {
      expect(error).toBeInstanceOf(ExaoneResponseError)
      expect(error).toBeInstanceOf(ModelError)
      expect((error as ExaoneResponseError).body).toEqual({ error: { message: '[REDACTED]', code: 'code-[REDACTED]', type: 'model_error' }, '[REDACTED]': 'secret' })
      expect((error as ExaoneResponseError).code).toBe('code-[REDACTED]')
      expect(String(error)).not.toContain('HTTP 200')
      expect(JSON.stringify(error)).not.toContain('private-key')
    }
  })

  it('accepts a null error field on successful responses', async () => {
    const data = sse([{ ...event({ content: 'ok' }, 'stop'), error: null }])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await aggregate(model)
    expect(result.stopReason).toBe('endTurn')
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
