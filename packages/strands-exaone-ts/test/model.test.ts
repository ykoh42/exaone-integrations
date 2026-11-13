import { inspect } from 'node:util'
import { describe, expect, it, vi } from 'vitest'
import { Agent, Message, ModelError, ModelNetworkError, ModelThrottledError, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneAPIError, ExaoneModel, ExaoneResponseError } from '../src/index.js'
import { readSSE } from '../src/sse.js'
import { redactMetadata } from '../src/metadata.js'

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
  it('keeps the API key out of object inspection and redacts encoded metadata values', () => {
    const apiKey = 'synthetic+private/key with spaces'
    const model = new ExaoneModel({ apiKey })
    const encoded = encodeURIComponent(apiKey)
    const formEncoded = new URLSearchParams({ value: apiKey }).toString().slice('value='.length)

    expect(JSON.stringify(model)).not.toContain(apiKey)
    expect(inspect(model)).not.toContain(apiKey)
    expect(JSON.stringify(model.getConfig())).not.toContain(apiKey)
    expect(redactMetadata({ [encoded]: formEncoded }, apiKey)).toEqual({ '[REDACTED]': '[REDACTED]' })
    expect(redactMetadata({ value: encoded.replace('%2B', '%2b') }, apiKey)).toEqual({ value: '[REDACTED]' })
    const differentValue = encoded.replace('%2F', '%3F')
    expect(redactMetadata({ [differentValue]: differentValue }, apiKey)).toEqual({ [differentValue]: differentValue })
  })

  it('redacts encoded API keys from raw HTTP errors, request IDs, and transport diagnostics', async () => {
    const apiKey = 'synthetic+private/key with spaces'
    const encoded = encodeURIComponent(apiKey).replace('%2B', '%2b')
    const httpModel = new ExaoneModel({
      apiKey,
      fetch: transport(response(`provider rejected ${encoded}`, 400, { 'x-request-id': encoded })),
    })
    try {
      await collect(httpModel)
      throw new Error('Expected HTTP error')
    } catch (error) {
      expect(error).toBeInstanceOf(ExaoneAPIError)
      expect((error as ExaoneAPIError).body).toBe('provider rejected [REDACTED]')
      expect((error as ExaoneAPIError).requestId).toBe('[REDACTED]')
    }

    const transportModel = new ExaoneModel({
      apiKey,
      fetch: async () => { throw new TypeError(`request failed ${encoded}`) },
    })
    try {
      await collect(transportModel)
      throw new Error('Expected network error')
    } catch (error) {
      expect((error as Error).cause).toMatchObject({ message: 'request failed [REDACTED]' })
    }
  })

  it('does not clone callback metadata when no callback is configured', async () => {
    const messages = [user()]
    const model = new ExaoneModel({
      apiKey: 'test-key',
      fetch: async () => response(sse([event({ content: 'answer' }, 'stop')]), 200, { 'content-type': 'text/event-stream' }),
    })
    const clone = vi.spyOn(globalThis, 'structuredClone')
    try {
      await collect(model, messages)
      expect(clone).not.toHaveBeenCalled()
    } finally {
      clone.mockRestore()
    }
  })

  it('validates named EXAONE options at construction and preserves configuration after a rejected update', () => {
    expect(() => new ExaoneModel({ topK: 0 })).toThrow('topK must be -1 or a positive integer')
    expect(() => new ExaoneModel({ topP: 1.01 })).toThrow('topP must be a finite number between 0 and 1')
    expect(() => new ExaoneModel({ temperature: -0.1 })).toThrow('temperature must be a finite number at least 0')
    expect(() => new ExaoneModel({ presencePenalty: 2.01 })).toThrow('presencePenalty must be a finite number between -2 and 2')
    expect(() => new ExaoneModel({ maxTokens: -1 })).toThrow('maxTokens must be a non-negative integer')
    expect(() => new ExaoneModel({ maxTokens: 524289 })).toThrow('maxTokens cannot exceed the EXAONE context limit')
    expect(() => new ExaoneModel({ contextWindowLimit: 0 })).toThrow('contextWindowLimit must be a positive integer')
    expect(() => new ExaoneModel({ reasoningEffort: 'extreme' as never })).toThrow('reasoningEffort must be one of')
    expect(() => new ExaoneModel({ includeUsage: 'yes' as never })).toThrow('includeUsage must be a boolean')
    expect(() => new ExaoneModel({ timeoutMs: 1.5 })).toThrow(RangeError)
    expect(() => new ExaoneModel({ modelInfo: { limits: { contextWindow: 0 } } })).toThrow('modelInfo.limits.contextWindow must be a positive integer')
    expect(() => new ExaoneModel({ params: { n: 2 } })).toThrow('Strands supports exactly one completion choice')
    expect(() => new ExaoneModel({ params: { stream_options: [] } })).toThrow('stream_options must be an object or null')

    const model = new ExaoneModel({ topK: 1, topP: 0, temperature: 0, presencePenalty: -2, maxTokens: 0 })
    expect(() => model.updateConfig({ topP: 1.01 })).toThrow('topP must be a finite number between 0 and 1')
    expect(model.getConfig()).toMatchObject({ topK: 1, topP: 0, temperature: 0, presencePenalty: -2, maxTokens: 0 })
  })

  it.each([
    'http://api.example/v1',
    'ftp://api.example/v1',
    'https://user:password@api.example/v1',
    'https://api.example/v1?token=secret',
    'https://api.example/v1#fragment',
  ])('rejects unsafe baseURL values: %s', (baseURL) => {
    expect(() => new ExaoneModel({ apiKey: 'key', baseURL })).toThrow(/baseURL/)
  })

  it.each(['localhost', '127.0.0.1', '[::1]'])('allows loopback HTTP for local test servers: %s', (host) => {
    const model = new ExaoneModel({ apiKey: 'key', baseURL: `http://${host}:8000/v1` })
    expect(model.getConfig().baseURL).toBe(`http://${host}:8000/v1`)
  })

  it('explains both ways to provide the API key', async () => {
    await expect(collect(new ExaoneModel({ apiKey: '' }))).rejects.toThrow(/apiKey.*EXAONE_API_KEY/)
  })

  it('uses modelInfo limits according to the Strands merge contract', () => {
    const model = new ExaoneModel({
      apiKey: 'key',
      contextWindowLimit: 4096,
      modelInfo: { limits: { contextWindow: 2048, maxOutputTokens: 1024 } },
    })
    expect(model.modelInfo.limits.contextWindow).toBe(2048)
    expect(model.modelInfo.limits.maxOutputTokens).toBe(1024)
    const capabilitiesOnly = new ExaoneModel({ modelInfo: { capabilities: { nativeStructuredOutput: true } } })
    expect(capabilitiesOnly.modelInfo.limits.contextWindow).toBe(524_288)
  })

  it.each(['low', 'medium', 'high', 'max'] as const)('sends reasoning effort %s', async (reasoningEffort) => {
    const fetcher = vi.fn(async () => response(sse([event({ content: 'ok' }), event({}, 'stop')])))
    await collect(new ExaoneModel({ apiKey: 'key', reasoningEffort, fetch: fetcher as typeof fetch }))
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    expect(JSON.parse(String(init.body))).toMatchObject({ reasoning_effort: reasoningEffort })
  })

  it('omits an unspecified reasoning effort', async () => {
    const fetcher = vi.fn(async () => response(sse([event({ content: 'ok' }), event({}, 'stop')])))
    await collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher as typeof fetch }))
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    expect(JSON.parse(String(init.body))).not.toHaveProperty('reasoning_effort')
  })

  it('preserves reasoning, text and usage while formatting request options', async () => {
    const data = sse([
      event({ reasoning_content: '생각' }),
      event({ content: ' 답' }),
      event({}, 'stop'),
      wire([], { usage, result_code: 'SUCCESS', description: 'Synthetic success metadata' }),
    ])
    const fetcher = vi.fn(async () => response(data, 200, { 'x-request-id': 'request-1' }))
    const onMetadata = vi.fn()
    const model = new ExaoneModel({ apiKey: 'private-key', fetch: fetcher as typeof fetch, reasoningEffort: 'medium', preserveThinking: true, enableThinking: false, topK: 1, maxTokens: 0, includeUsage: false, onMetadata })
    const result = await aggregate(model)
    expect(result.message.content.map((b) => b.type)).toEqual(['reasoningBlock', 'textBlock'])
    expect(result.message.content[0]).toMatchObject({ text: '생각' })
    expect(result.message.content[1]).toMatchObject({ text: ' 답' })
    expect(result.metadata?.usage).toEqual({ inputTokens: 3, outputTokens: 5, totalTokens: 8 })
    expect(result.metadata?.trace).toMatchObject({ exaone: { requestId: 'request-1', usage: { reasoning_tokens: 2 } } })
    expect(onMetadata).toHaveBeenCalledOnce()
    expect(onMetadata.mock.calls[0]![0]).toMatchObject({
      status: 200,
      response: { result_code: 'SUCCESS', description: 'Synthetic success metadata' },
    })
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body).toMatchObject({ model: 'k-exaone_v2.5', max_tokens: 0, top_k: 1, reasoning_effort: 'medium', stream_options: { include_usage: false }, chat_template_kwargs: { preserve_thinking: true, enable_thinking: false } })
    expect(JSON.stringify(model.getConfig())).not.toContain('private-key')
  })

  it('passes through additional Chat Completions parameters and stream options', async () => {
    const fetcher = vi.fn(async () => response(sse([event({ content: 'ok' }), event({}, 'stop')])))
    const model = new ExaoneModel({
      apiKey: 'key',
      includeUsage: true,
      params: {
        frequency_penalty: 0.25,
        seed: 0,
        stop: ['<END>'],
        logprobs: true,
        top_logprobs: 3,
        parallel_tool_calls: false,
        response_format: { type: 'json_object' },
        stream_options: { include_usage: false, custom_extension: 'retained' },
      },
      fetch: fetcher as typeof fetch,
    })
    await collect(model)
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body).toMatchObject({
      frequency_penalty: 0.25,
      seed: 0,
      stop: ['<END>'],
      logprobs: true,
      top_logprobs: 3,
      parallel_tool_calls: false,
      response_format: { type: 'json_object' },
      stream_options: { include_usage: true, custom_extension: 'retained' },
    })
  })

  it.each([
    ['length', 'maxTokens'],
    ['content_filter', 'contentFiltered'],
  ] as const)('maps finish_reason=%s into the Strands stop reason', async (finish, expected) => {
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(sse([event({ content: 'partial' }), event({}, finish)]))) })
    const events = await collect(model)
    expect(events.find((item) => item.type === 'modelMessageStopEvent')).toMatchObject({ stopReason: expected })
  })

  it.each([true, false])('exposes refusal-only responses as assistant text (stream=%s)', async (streaming) => {
    const refusal = 'I cannot help with that request.'
    const fetcher = transport(streaming
      ? response(sse([event({ refusal }), event({}, 'stop')]))
      : response(JSON.stringify(wire([{ index: 0, message: { refusal }, finish_reason: 'stop' }]))))
    const result = await aggregate(new ExaoneModel({ apiKey: 'key', stream: streaming, fetch: fetcher }))
    expect(result.message.content).toContainEqual(expect.objectContaining({ type: 'textBlock', text: refusal }))
  })

  it.each([true, false])('preserves refusal when content contains only whitespace (stream=%s)', async (streaming) => {
    const refusal = 'I cannot help with that request.'
    const fetcher = transport(streaming
      ? response(sse([event({ content: '\n\n', refusal }), event({}, 'stop')]))
      : response(JSON.stringify(wire([{ index: 0, message: { content: '\n\n', refusal }, finish_reason: 'stop' }]))))
    const result = await aggregate(new ExaoneModel({ apiKey: 'key', stream: streaming, fetch: fetcher }))
    const text = result.message.content.filter((block) => block.type === 'textBlock').map((block) => block.text).join('')
    expect(text).toBe(`\n\n${refusal}`)
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
      Message.fromMessageData({ role: 'user', content: [{ text: 'next' }, { toolResult: { toolUseId: 'id1', status: 'success', content: [{ text: '7' }] } }] }),
    ]
    await collect(model, messages)
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body.messages).toEqual([
      { role: 'assistant', content: null, reasoning_content: 'reason', tool_calls: [{ id: 'id1', type: 'function', function: { name: 'tool', arguments: '{"x":2}' } }] },
      { role: 'tool', tool_call_id: 'id1', content: '7' }, { role: 'user', content: 'next' },
    ])
  })

  it('marks error results while preserving empty, text, JSON, and successful content', async () => {
    const fetcher = vi.fn(async () => response(JSON.stringify(wire([{ index: 0, message: { content: 'ok' }, finish_reason: 'stop' }]))))
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher as typeof fetch, stream: false })
    const blocks = [
      { toolUseId: 'empty-error', status: 'error' as const, content: [] },
      { toolUseId: 'text-error', status: 'error' as const, content: [{ text: '\n failure \n' }] },
      { toolUseId: 'json-error', status: 'error' as const, content: [{ json: { value: 0, ok: false } }] },
      { toolUseId: 'mixed-error', status: 'error' as const, content: [{ text: 'before' }, { json: { value: 1 } }, { text: 'after' }] },
      { toolUseId: 'empty-success', status: 'success' as const, content: [] },
      { toolUseId: 'text-success', status: 'success' as const, content: [{ text: '\n failure \n' }] },
      { toolUseId: 'json-success', status: 'success' as const, content: [{ json: { value: 0, ok: false } }] },
    ]
    await collect(model, [Message.fromMessageData({ role: 'user', content: blocks.map(toolResult => ({ toolResult })) })])
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    expect(JSON.parse(String(init.body)).messages).toEqual([
      { role: 'tool', tool_call_id: 'empty-error', content: '[ERROR] ' },
      { role: 'tool', tool_call_id: 'text-error', content: '[ERROR] \n failure \n' },
      { role: 'tool', tool_call_id: 'json-error', content: '[ERROR] {"value":0,"ok":false}' },
      { role: 'tool', tool_call_id: 'mixed-error', content: '[ERROR] before\n{"value":1}\nafter' },
      { role: 'tool', tool_call_id: 'empty-success', content: '' },
      { role: 'tool', tool_call_id: 'text-success', content: '\n failure \n' },
      { role: 'tool', tool_call_id: 'json-success', content: '{"value":0,"ok":false}' },
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
      expect((error as ExaoneResponseError).errorType).toBe('model_error')
      expect((error as ExaoneResponseError).code).toBe('code-[REDACTED]')
      expect((error as Error).message).toBe('[REDACTED]')
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

  it.each(['{', '{"query":', '{"query":"unterminated', 'null', '42', '[]'])('rejects incomplete or non-object tool arguments %s', async (args) => {
    const data = sse([event({ tool_calls: [{ index: 0, id: 'id', function: { name: 'fn', arguments: args } }] }), event({}, 'tool_calls')])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    await expect(collect(model)).rejects.toThrow(ModelError)
  })

  it.each([
    ['{"query":"EXAONE"', '{"query":"EXAONE"}'],
    ['{"query":"EXAONE","options":{"limit":2', '{"query":"EXAONE","options":{"limit":2}}'],
    ['{"query":"brace } inside string"', '{"query":"brace } inside string"}'],
    ['{"query":"escaped \\" } quote"', '{"query":"escaped \\" } quote"}'],
    ['{"items":[1,2', '{"items":[1,2]}'],
    ['{"query":"EXAONE",}', '{"query":"EXAONE"}'],
    ['{"items":[1,2,]}', '{"items":[1,2]}'],
  ])('repairs tool arguments without changing their values: %s', async (raw, expected) => {
    const data = sse([event({ tool_calls: [{ index: 0, id: 'id', function: { name: 'fn', arguments: raw } }] }), event({}, 'tool_calls')])
    const model = new ExaoneModel({ apiKey: 'key', fetch: transport(response(data)) })
    const result = await collect(model)
    const toolInput = result.find((item) => item.type === 'modelContentBlockDeltaEvent' && item.delta.type === 'toolUseInputDelta')
    expect(toolInput).toMatchObject({ delta: { input: expected } })
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
      expect(cause.errorType).toBe('rate_limit_error')
      expect(cause.code).toBe('rate_limit_exceeded')
      expect(cause.param).toBeUndefined()
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

  it('exposes diagnostics from both EXAONE error envelope shapes', async () => {
    const openAIShape = new ExaoneModel({
      apiKey: 'key',
      fetch: transport(response(JSON.stringify({ error: { type: 'invalid_request_error', code: null, param: 'top_k', message: 'invalid value' } }), 400, { 'x-request-id': 'request-openai' })),
    })
    await expect(collect(openAIShape)).rejects.toMatchObject({
      message: 'invalid value',
      status: 400,
      errorType: 'invalid_request_error',
      code: null,
      param: 'top_k',
      requestId: 'request-openai',
    })

    const documentedShape = new ExaoneModel({
      apiKey: 'key',
      fetch: transport(response(JSON.stringify({ result_code: 'INVALID_INPUT', description: 'invalid request' }), 422, { 'x-request-id': 'request-documented' })),
    })
    await expect(collect(documentedShape)).rejects.toMatchObject({
      message: 'invalid request',
      status: 422,
      errorType: undefined,
      code: 'INVALID_INPUT',
      param: undefined,
      requestId: 'request-documented',
      body: { result_code: 'INVALID_INPUT', description: 'invalid request' },
    })
  })

  it('formats structured output without overriding forced tool calls', async () => {
    const fetcher = vi.fn(async () => response(JSON.stringify(wire([{ index: 0, message: { content: '{"x":1}' }, finish_reason: 'stop' }]))))
    const model = new ExaoneModel({ apiKey: 'key', stream: false, fetch: fetcher as typeof fetch })
    for await (const _ of model.stream([user()], { structuredOutputSchema: z.object({ x: z.number() }) })) { /* consume */ }
    const init = (fetcher.mock.calls as unknown as [string, RequestInit][])[0]![1]
    const body = JSON.parse(String(init.body))
    expect(body.response_format).toMatchObject({ type: 'json_schema', json_schema: { strict: true } })
  })

  it('rejects multiple requested choices during configuration validation', () => {
    expect(() => new ExaoneModel({ apiKey: 'key', params: { n: 2 } })).toThrow('Strands supports exactly one completion choice')
  })
})

describe('SSE parser', () => {
  it.each([1, 7, 1024])('preserves a large Unicode event split into %i-byte fragments', async (fragment) => {
    const content = '한a'.repeat(40000)
    const bytes = new TextEncoder().encode(`\uFEFFdata: ${content}\r\ndata: tail\r\n\r\n`)
    let offset = 0
    const stream = new ReadableStream<Uint8Array>({ pull(controller) {
      if (offset >= bytes.length) return controller.close()
      controller.enqueue(bytes.subarray(offset, Math.min(offset + fragment, bytes.length)))
      offset += fragment
    } })
    const events = []
    for await (const value of readSSE(stream)) events.push(value)
    expect(events).toEqual([`${content}\ntail`])
  })

  it.each(['\n', '\r', '\r\n'])('strips a split leading BOM, dispatches blank-line events and discards the incomplete tail (%j)', async (newline) => {
    const bytes = new TextEncoder().encode(`\uFEFFdata: 안녕${newline}${newline}data: pending${newline}`)
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
      controller.close()
    } })
    const events = []
    for await (const value of readSSE(stream)) events.push(value)
    expect(events).toEqual(['안녕'])
  })

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
