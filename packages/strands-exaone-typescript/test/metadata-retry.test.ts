import { afterEach, describe, expect, it, vi } from 'vitest'
import { Message, ModelError, ModelNetworkError, ModelThrottledError } from '@strands-agents/sdk'
import { ExaoneAPIError, ExaoneModel } from '../src/index.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const ok = () => new Response(JSON.stringify({ choices: [{ index: 0, message: { content: 'ok' }, finish_reason: 'stop' }] }))
const stream = (events: unknown[]) => new Response(events.map((item) => `data: ${JSON.stringify(item)}\n\n`).join('') + 'data: [DONE]\n\n')
const fakeFetch = (fn: (...args: Parameters<typeof fetch>) => Promise<Response>) => vi.fn(fn) as unknown as typeof fetch
async function collect(model: ExaoneModel, signal?: AbortSignal) {
  const events = []
  for await (const item of model.stream([user()], { cancelSignal: signal })) events.push(item)
  return events
}

afterEach(() => vi.useRealTimers())

describe('response metadata', () => {
  const tokenFields = ['prompt_tokens', 'completion_tokens', 'total_tokens'] as const
  const malformedCounts = [
    { name: 'negative', wire: -1, expected: -1 },
    { name: 'fractional', wire: 1.5, expected: 1.5 },
    { name: 'unsafe integer', wire: Number.MAX_SAFE_INTEGER + 1, expected: Number.MAX_SAFE_INTEGER + 1 },
    { name: 'overflow', wire: '__overflow__', expected: Infinity },
  ]
  for (const field of tokenFields) {
    it.each(malformedCounts)(`omits malformed SDK usage for ${field} ($name) while preserving raw metadata`, async ({ wire, expected }) => {
      const usage = { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3, [field]: wire }
      const payload = [
        { choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] },
        { choices: [], usage },
      ].map((item) => `data: ${JSON.stringify(item).replace('"__overflow__"', '1e309')}\n\n`).join('') + 'data: [DONE]\n\n'
      const onMetadata = vi.fn()
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: async () => new Response(payload), onMetadata })
      const events = await collect(model)
      expect(events).toContainEqual(expect.objectContaining({ type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'ok' } }))
      expect(events.find((item) => item.type === 'modelMetadataEvent')).not.toHaveProperty('usage')
      expect(onMetadata).toHaveBeenCalledOnce()
      expect(onMetadata.mock.calls[0]![0].usage?.[field]).toBe(expected)
    })
  }

  it('retains zero token counts in SDK usage and raw metadata', async () => {
    const onMetadata = vi.fn()
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: async () => stream([
      { choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } },
    ]), onMetadata })
    const events = await collect(model)
    expect(events.find((item) => item.type === 'modelMetadataEvent')).toMatchObject({ usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } })
    expect(onMetadata.mock.calls[0]![0].usage).toMatchObject({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  })

  it('requests streaming usage by default, respects updates, and omits stream options for non-streaming requests', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher = fakeFetch(async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      return request.stream ? stream([{ choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] }]) : ok()
    })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    await collect(model)
    model.updateConfig({ includeUsage: false })
    await collect(model)
    model.updateConfig({ includeUsage: true, stream: false })
    await collect(model)
    expect(requests[0]).toMatchObject({ stream: true, stream_options: { include_usage: true } })
    expect(requests[1]).toMatchObject({ stream: true, stream_options: { include_usage: false } })
    expect(requests[2]).toMatchObject({ stream: false })
    expect(requests[2]).not.toHaveProperty('stream_options')
  })

  it('reports abort as an error and redacts credentials from unknown finish reasons', async () => {
    const aborted = fakeFetch(async () => stream([{ choices: [{ index: 0, delta: {}, finish_reason: 'abort' }] }]))
    await expect(collect(new ExaoneModel({ apiKey: 'private-secret', fetch: aborted }))).rejects.toThrow('abort')
    const unrecognized = fakeFetch(async () => stream([{ choices: [{ index: 0, delta: {}, finish_reason: 'private-secret' }] }]))
    await expect(collect(new ExaoneModel({ apiKey: 'private-secret', fetch: unrecognized }))).rejects.toThrow('[REDACTED]')
  })

  it('accumulates stream metadata and all logprob fragments, redacting credentials', async () => {
    const onMetadata = vi.fn()
    const chunks = [
      {
        id: 'resp-private-secret', object: 'chat.completion.chunk', model: 'example', created: 123,
        metadata: { version: 'private-secret' }, service_tier: null,
        choices: [{
          index: 0,
          delta: {
            reasoning_content: 'hidden-reasoning', refusal: 'Cannot ', annotations: [{ id: 1 }],
            audio: { data: 'fragment-a' }, function_call: { name: 'f' },
          },
          logprobs: { content: [{ token: 'one' }], refusal: null }, finish_reason: null,
        }],
      },
      {
        id: null,
        choices: [{
          index: 0,
          delta: {
            content: 'hidden-response', refusal: 'do that.', annotations: [{ id: 2 }],
            audio: { data: 'fragment-b-private-secret' }, function_call: { arguments: '{}' },
          },
          logprobs: { content: [{ token: 'two-private-secret' }] }, matched_stop: 'end', finish_reason: null,
        }],
      },
      { system_fingerprint: null, choices: [{ index: 0, delta: {}, finish_reason: 'stop', logprobs: null }] },
      { usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3, opaque: 'private-secret' }, choices: [] },
    ]
    const fetcher = fakeFetch(async () => stream(chunks))
    const model = new ExaoneModel({ apiKey: 'private-secret', fetch: fetcher, onMetadata })
    const events = await collect(model)
    expect(onMetadata).toHaveBeenCalledOnce()
    const metadata = onMetadata.mock.calls[0]![0]
    expect(metadata.response).toEqual({
      id: 'resp-[REDACTED]', object: 'chat.completion.chunk', model: 'example', created: 123,
      metadata: { version: '[REDACTED]' }, service_tier: null, system_fingerprint: null,
      choices: [{
        index: 0,
        finish_reason: 'stop',
        matched_stop: 'end',
        logprobs: { content: [{ token: 'one' }, { token: 'two-[REDACTED]' }], refusal: null },
        message: { refusal: 'Cannot do that.', annotations: [{ id: 1 }, { id: 2 }] },
        message_deltas: [
          { audio: { data: 'fragment-a' }, function_call: { name: 'f' } },
          { audio: { data: 'fragment-b-[REDACTED]' }, function_call: { arguments: '{}' } },
        ],
      }],
    })
    expect(metadata.usage.opaque).toBe('[REDACTED]')
    expect(JSON.stringify(metadata)).not.toContain('hidden-reasoning')
    expect(JSON.stringify(metadata)).not.toContain('hidden-response')
    expect(JSON.stringify(metadata)).not.toContain('private-secret')
    const metaEvent = events.find((item) => item.type === 'modelMetadataEvent')
    expect(metaEvent?.trace).toMatchObject({ exaone: metadata })
  })

  it('preserves non-streaming fields and masks secrets in headers and nested metadata', async () => {
    const onMetadata = vi.fn()
    const payload = {
      id: 'id-1', metadata: { 'key-private-secret': 'private-secret' }, moderation: { result: true },
      service_tier: null,
      choices: [{
        index: 0,
        message: { content: 'answer', refusal: 'private-secret', annotations: [], audio: { id: 'audio-id' }, function_call: null },
        finish_reason: 'stop', logprobs: { content: [{ token: 'token' }] }, matched_stop: null,
      }],
    }
    const fetcher = fakeFetch(async () => new Response(JSON.stringify(payload), { headers: { 'x-request-id': 'private-secret', 'retry-after': 'private-secret' } }))
    await collect(new ExaoneModel({ apiKey: 'private-secret', stream: false, fetch: fetcher, onMetadata }))
    expect(onMetadata.mock.calls[0]![0]).toMatchObject({
      requestId: '[REDACTED]', rateLimit: { 'retry-after': '[REDACTED]' },
      response: {
        id: 'id-1', moderation: { result: true }, service_tier: null,
        metadata: { 'key-[REDACTED]': '[REDACTED]' },
        choices: [{
          index: 0,
          finish_reason: 'stop',
          logprobs: { content: [{ token: 'token' }] },
          matched_stop: null,
          message: { refusal: '[REDACTED]', annotations: [], audio: { id: 'audio-id' }, function_call: null },
        }],
      },
    })
  })
})

describe('opt-in retry policy', () => {
  it('waits for a server delay longer than the Node timer limit', async () => {
    vi.useFakeTimers()
    const delay = 2_147_483_648
    let attempts = 0
    const fetcher = fakeFetch(async () => ++attempts === 1
      ? new Response('{}', { status: 429, headers: { 'retry-after-ms': String(delay) } })
      : ok())
    const result = collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, stream: false, maxRetries: 1, maxRetryDelayMs: delay }))
    await vi.advanceTimersByTimeAsync(delay - 1)
    expect(attempts).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    await result
    expect(attempts).toBe(2)
  })

  it('does not round a fractional millisecond retry hint down', async () => {
    vi.useFakeTimers()
    let attempts = 0
    const fetcher = fakeFetch(async () => ++attempts === 1
      ? new Response('{}', { status: 429, headers: { 'retry-after-ms': '19.5' } })
      : ok())
    const result = collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, stream: false, maxRetries: 1 }))
    await vi.advanceTimersByTimeAsync(19)
    expect(attempts).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    await result
    expect(attempts).toBe(2)
  })

  it.each([0, 2_147_483_647])('cancels a long retry wait after %i ms without another request', async (elapsed) => {
    vi.useFakeTimers()
    const delay = 2_147_483_657
    const fetcher = fakeFetch(async () => new Response('{}', { status: 429, headers: { 'retry-after-ms': String(delay) } }))
    const signal = new AbortController()
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 1, maxRetryDelayMs: delay })
    const result = collect(model, signal.signal)
    const failure = expect(result).rejects.toThrow('cancelled')
    await vi.advanceTimersByTimeAsync(elapsed)
    signal.abort(new Error('cancelled'))
    await failure
    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('honors the longer of retry headers before emitting any result', async () => {
    vi.useFakeTimers()
    const cancel = vi.fn()
    let attempts = 0
    const fetcher = fakeFetch(async () => {
      attempts++
      if (attempts > 1) return ok()
      return new Response(new ReadableStream({ cancel }), { status: 429, headers: { 'retry-after': '0.01', 'retry-after-ms': '20' } })
    })
    const model = new ExaoneModel({ apiKey: 'key', stream: false, maxRetries: 1, maxRetryDelayMs: 100, fetch: fetcher })
    const result = collect(model)
    await vi.advanceTimersByTimeAsync(19)
    expect(attempts).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(attempts).toBe(2)
    expect((await result).some((e) => e.type === 'modelMessageStopEvent')).toBe(true)
    expect(cancel).toHaveBeenCalledOnce()
  })

  it.each([
    { retryAfter: '10', retryAfterMs: '9598', delay: 10_000 },
    { retryAfter: '16', retryAfterMs: '15659', delay: 16_000 },
  ])('honors captured EXAONE retry headers for at least $delay ms', async ({ retryAfter, retryAfterMs, delay }) => {
    vi.useFakeTimers()
    let attempts = 0
    const fetcher = fakeFetch(async () => ++attempts === 1
      ? new Response('{}', {
        status: 429,
        headers: {
          'retry-after': retryAfter,
          'retry-after-ms': retryAfterMs,
          'x-second-rate-limit': '1',
          'x-second-rate-limit-remaining': '1',
          'x-minute-rate-limit': '60',
          'x-minute-rate-limit-remaining': '60',
          'x-day-rate-limit': '86400',
          'x-day-rate-limit-remaining': '86086',
        },
      })
      : ok())
    const result = collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, stream: false, maxRetries: 1 }))
    await vi.advanceTimersByTimeAsync(delay - 1)
    expect(attempts).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    await result
    expect(attempts).toBe(2)
  })

  it('uses bounded backoff when captured 429 responses omit retry hints', async () => {
    vi.useFakeTimers()
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.5)
    let attempts = 0
    const fetcher = fakeFetch(async () => ++attempts === 1
      ? new Response('{}', {
        status: 429,
        headers: {
          'x-second-rate-limit': '1',
          'x-second-rate-limit-remaining': '1',
          'x-minute-rate-limit': '60',
          'x-minute-rate-limit-remaining': '60',
          'x-day-rate-limit': '86400',
          'x-day-rate-limit-remaining': '86086',
        },
      })
      : ok())
    try {
      const result = collect(new ExaoneModel({
        apiKey: 'key', fetch: fetcher, stream: false, maxRetries: 1, baseRetryDelayMs: 1_000,
      }))
      await vi.advanceTimersByTimeAsync(499)
      expect(attempts).toBe(1)
      await vi.advanceTimersByTimeAsync(1)
      await result
      expect(attempts).toBe(2)
    } finally {
      random.mockRestore()
    }
  })

  it('does not shorten a server retry delay beyond the configured cap', async () => {
    const fetcher = fakeFetch(async () => new Response('{"error":{"message":"limit"}}', { status: 429, headers: { 'retry-after-ms': '1000' } }))
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 3, maxRetryDelayMs: 10 })
    await expect(collect(model)).rejects.toBeInstanceOf(ModelThrottledError)
    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('redacts the credential from retry errors, including response headers', async () => {
    const fetcher = fakeFetch(async () => new Response('{"error":{"message":"private-secret"}}', { status: 429, headers: { 'x-request-id': 'private-secret', 'retry-after': 'private-secret' } }))
    try {
      await collect(new ExaoneModel({ apiKey: 'private-secret', fetch: fetcher }))
      throw new Error('expected an HTTP error')
    } catch (error) {
      expect(error).toBeInstanceOf(ModelThrottledError)
      const cause = (error as Error).cause as ExaoneAPIError
      expect(cause.requestId).toBe('[REDACTED]')
      expect(cause.headers['retry-after']).toBe('[REDACTED]')
      expect(JSON.stringify(cause.body)).not.toContain('private-secret')
    }
  })

  it('redacts JSON-escaped credential values in error bodies', async () => {
    const key = 'secret"with\\escape'
    const fetcher = fakeFetch(async () => new Response(JSON.stringify({ error: { message: key } }), { status: 400 }))
    try {
      await collect(new ExaoneModel({ apiKey: key, fetch: fetcher }))
      throw new Error('expected an HTTP error')
    } catch (error) {
      expect(error).toBeInstanceOf(ExaoneAPIError)
      expect((error as ExaoneAPIError).body).toEqual({ error: { message: '[REDACTED]' } })
    }
  })

  it('accepts HTTP-date retry hints and uses the longer delay', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-11-11T00:00:00.000Z'))
    let attempts = 0
    const fetcher = fakeFetch(async () => ++attempts === 1
      ? new Response('{}', { status: 503, headers: { 'retry-after': 'Wed, 11 Nov 2026 00:00:03 GMT', 'retry-after-ms': '1000' } })
      : ok())
    const result = collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, stream: false, maxRetries: 1 }))
    await vi.advanceTimersByTimeAsync(2999)
    expect(attempts).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    await result
    expect(attempts).toBe(2)
  })

  it('does not retry client errors and stops at the configured attempt limit', async () => {
    const badRequest = fakeFetch(async () => new Response('{}', { status: 400 }))
    await expect(collect(new ExaoneModel({ apiKey: 'key', fetch: badRequest, maxRetries: 2 }))).rejects.toBeInstanceOf(ExaoneAPIError)
    expect(badRequest).toHaveBeenCalledOnce()
    const server = fakeFetch(async () => new Response('{}', { status: 503 }))
    await expect(collect(new ExaoneModel({ apiKey: 'key', fetch: server, maxRetries: 2, baseRetryDelayMs: 0 }))).rejects.toBeInstanceOf(ExaoneAPIError)
    expect(server).toHaveBeenCalledTimes(3)
  })

  it('does not retry ambiguous fetch failures that may occur after sending', async () => {
    const fetcher = fakeFetch(async () => { throw new TypeError('socket closed after sending') })
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 2, baseRetryDelayMs: 0 })
    await expect(collect(model)).rejects.toBeInstanceOf(ModelNetworkError)
    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('does not retry failures after partial stream output', async () => {
    const encoder = new TextEncoder()
    let sent = false
    const broken = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (!sent) {
          sent = true
          controller.enqueue(encoder.encode('data: {"choices":[{"index":0,"delta":{"content":"part"}}]}\n\n'))
        } else controller.error(new TypeError('connection lost'))
      },
    })
    const partial = fakeFetch(async () => new Response(broken))
    const events = []
    const model = new ExaoneModel({ apiKey: 'key', fetch: partial, maxRetries: 2, baseRetryDelayMs: 0 })
    const consume = async () => { for await (const item of model.stream([user()])) events.push(item) }
    await expect(consume()).rejects.toBeInstanceOf(ModelError)
    expect(events).toContainEqual({ type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'part' } })
    expect(partial).toHaveBeenCalledOnce()
  })

  it('redacts fetch error diagnostics without exposing the original nested cause or stack', async () => {
    const secret = 'private-secret'
    const original = new TypeError(`socket ${secret}`, { cause: new Error(`inner ${secret}`) })
    const fetcher = fakeFetch(async () => { throw original })
    try {
      await collect(new ExaoneModel({ apiKey: secret, fetch: fetcher }))
      throw new Error('expected a network error')
    } catch (error) {
      expect(error).toBeInstanceOf(ModelNetworkError)
      const cause = (error as Error).cause as Error
      expect(cause).not.toBe(original)
      expect(cause).toBeInstanceOf(TypeError)
      expect(cause.message).toBe('socket [REDACTED]')
      expect(cause.cause).toBeUndefined()
      expect(cause.stack).not.toContain(secret)
    }
  })

  it('redacts transport diagnostics on response stream failure', async () => {
    const secret = 'private-secret'
    const original = new TypeError(`response ${secret}`, { cause: new Error(`inner ${secret}`) })
    const broken = new ReadableStream<Uint8Array>({ start(controller) { controller.error(original) } })
    const fetcher = fakeFetch(async () => new Response(broken))
    try {
      await collect(new ExaoneModel({ apiKey: secret, fetch: fetcher }))
      throw new Error('expected a network error')
    } catch (error) {
      expect(error).toBeInstanceOf(ModelNetworkError)
      const cause = (error as Error).cause as Error
      expect(cause).not.toBe(original)
      expect(cause.message).toBe('EXAONE response body read failed')
      expect(cause.cause).toBeUndefined()
      expect(cause.stack).not.toContain(secret)
    }
  })

  it('continues retrying HTTP responses when body cancellation rejects', async () => {
    let attempts = 0
    const fetcher = fakeFetch(async () => {
      if (++attempts > 1) return ok()
      const body = new ReadableStream({ cancel() { throw new Error('cancel failed') } })
      return new Response(body, { status: 503, headers: { 'retry-after': '0' } })
    })
    const result = await collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 1, stream: false }))
    expect(result.some((event) => event.type === 'modelMessageStopEvent')).toBe(true)
    expect(attempts).toBe(2)
  })

  it('allows cancellation during retry wait even when the previous body never cancels', async () => {
    vi.useFakeTimers()
    const cancel = vi.fn(() => new Promise<void>(() => undefined))
    const fetcher = fakeFetch(async () => new Response(new ReadableStream({ cancel }), {
      status: 503, headers: { 'retry-after': '20' },
    }))
    const signal = new AbortController()
    const pending = collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 1 }), signal.signal)
    const failure = expect(pending).rejects.toThrow('cancelled')
    await vi.advanceTimersByTimeAsync(1)
    expect(cancel).toHaveBeenCalledOnce()
    signal.abort(new Error('cancelled'))
    await failure
    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('continues retrying when the previous body cancellation never settles', async () => {
    let attempts = 0
    const fetcher = fakeFetch(async () => {
      if (++attempts > 1) return ok()
      const body = new ReadableStream({ cancel: () => new Promise<void>(() => undefined) })
      return new Response(body, { status: 503, headers: { 'retry-after': '0' } })
    })
    const result = await collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 1, stream: false }))
    expect(result.some((event) => event.type === 'modelMessageStopEvent')).toBe(true)
    expect(attempts).toBe(2)
  })

  it('cancels pending backoff without another request', async () => {
    vi.useFakeTimers()
    const fetcher = fakeFetch(async () => new Response('{}', { status: 429, headers: { 'retry-after': '20' } }))
    const signal = new AbortController()
    const pending = collect(new ExaoneModel({ apiKey: 'key', fetch: fetcher, maxRetries: 2 }), signal.signal)
    const failure = expect(pending).rejects.toThrow('cancelled')
    await vi.advanceTimersByTimeAsync(1)
    signal.abort(new Error('cancelled'))
    await failure
    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('validates retry settings on construction and atomically on update', () => {
    for (const config of [{ maxRetries: -1 }, { maxRetries: 1.2 }, { baseRetryDelayMs: -1 }, { maxRetryDelayMs: Infinity }]) {
      expect(() => new ExaoneModel(config)).toThrow(TypeError)
    }
    const model = new ExaoneModel({ maxRetries: 2 })
    expect(() => model.updateConfig({ maxRetries: -1 })).toThrow(TypeError)
    expect(model.getConfig().maxRetries).toBe(2)
  })
})
