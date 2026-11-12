import { createHarness } from '@strands-agents/harness'
import { DefaultModelRetryStrategy, Message, ModelError, ModelNetworkError, ModelThrottledError } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { ExaoneAPIError, ExaoneModel } from '../src/index.js'

const key = 'synthetic-secret'
const privatePrompt = 'synthetic-private-prompt'
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }
const types = [Error, TypeError, SyntaxError] as const
const chunk = (delta: unknown) => ({ choices: [{ index: 0, delta, finish_reason: null }] })
const broken = (error: Error, partial = false) => new ReadableStream<Uint8Array>({
  start(controller) {
    if (partial) controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk({ content: 'partial' }))}\n\n`))
    else controller.error(error)
  },
  pull(controller) { if (partial) controller.error(error) },
})
const originalError = (ErrorType: typeof Error) => {
  const error = new ErrorType(`read failed: ${key}, ${privatePrompt}`, { cause: new Error(`cause: ${key}, ${privatePrompt}`) })
  Object.assign(error, { request: { key, prompt: privatePrompt } })
  return error
}
function expectSafe(error: unknown) {
  const first = error as Error
  const serialized = [first.message, first.stack, (first.cause as Error | undefined)?.message, (first.cause as Error | undefined)?.stack].join('\n')
  expect(serialized).not.toContain(key)
  expect(serialized).not.toContain(privatePrompt)
}

/** Synthetic transports only; these are not observed EXAONE failures. */
describe('response body read failures through Strands harness', () => {
  it.each([
    { name: 'string', value: `${key} ${privatePrompt}` },
    { name: 'object', value: { message: `${key} ${privatePrompt}`, request: { key, privatePrompt } } },
  ])('sanitizes a non-Error $name thrown while reading a body', async ({ value }) => {
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.error(value) } })
    const model = new ExaoneModel({ apiKey: key, fetch: async () => new Response(body) })
    const agent = await createHarness({ ...options, model })
    let failure: unknown
    try { await agent.invoke(privatePrompt) } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelNetworkError)
    expectSafe(failure)
    expect(((failure as Error).cause as Error).cause).toBeUndefined()
  })

  it('does not reclassify a user metadata callback exception as a read failure', async () => {
    const original = new Error('synthetic callback failure')
    const fetcher: typeof fetch = async () => new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
    const model = new ExaoneModel({ apiKey: key, fetch: fetcher, onMetadata: () => { throw original } })
    const agent = await createHarness({ ...options, model })
    let failure: unknown
    try { await agent.invoke('hello') } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelError)
    expect(failure).not.toBeInstanceOf(ModelNetworkError)
    expect((failure as Error).cause).toBe(original)
  })

  it('allows an SDK retry after a generic read failure before output', async () => {
    let attempts = 0
    const fetcher: typeof fetch = async () => {
      attempts++
      if (attempts === 1) return new Response(broken(originalError(Error)))
      return new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
    }
    const model = new ExaoneModel({ apiKey: key, fetch: fetcher })
    const retryStrategy = new DefaultModelRetryStrategy({ maxAttempts: 2, backoff: { nextDelay: () => 0 } })
    const agent = await createHarness({ ...options, model, retryStrategy })
    const result = await agent.invoke('hello')
    expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'ok' }))
    expect(attempts).toBe(2)
  })

  it('does not let the SDK default retry a generic read failure after output', async () => {
    const fetcher = vi.fn(async () => new Response(broken(originalError(Error), true)))
    const model = new ExaoneModel({ apiKey: key, fetch: fetcher })
    const agent = await createHarness({ model, builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false })
    let failure: unknown
    try { await agent.invoke(privatePrompt) } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelError)
    expect(failure).not.toBeInstanceOf(ModelNetworkError)
    expectSafe(failure)
    expect(fetcher).toHaveBeenCalledOnce()
  })

  for (const status of [400, 429, 500]) {
    it.each(types)(`safely reports a %s while reading an HTTP ${status} body`, async (ErrorType) => {
      const original = originalError(ErrorType)
      const fetcher = vi.fn(async () => new Response(broken(original), { status, headers: { 'x-request-id': `req-${key}`, 'retry-after': '0' } }))
      const model = new ExaoneModel({ apiKey: key, fetch: fetcher })
      const agent = await createHarness({ ...options, model })
      let failure: unknown
      try { await agent.invoke(privatePrompt) } catch (error) { failure = error }
      if (status === 429) expect(failure).toBeInstanceOf(ModelThrottledError)
      else expect(failure).toBeInstanceOf(ExaoneAPIError)
      const apiError = (status === 429 ? (failure as Error).cause : failure) as ExaoneAPIError
      expect(apiError.status).toBe(status)
      expect(apiError.requestId).toBe('req-[REDACTED]')
      expect(apiError.body).toBeUndefined()
      expect(apiError.cause).toBeInstanceOf(Error)
      expect(apiError.cause).not.toBe(original)
      expect((apiError.cause as Error).cause).toBeUndefined()
      expectSafe(apiError)
      expectSafe(failure)
      expect(fetcher).toHaveBeenCalledOnce()
    })
  }

  for (const partial of [false, true]) {
    it.each(types)(`safely classifies a %s stream read error after partial=${partial}`, async (ErrorType) => {
      const original = originalError(ErrorType)
      const fetcher = vi.fn(async () => new Response(broken(original, partial)))
      const model = new ExaoneModel({ apiKey: key, fetch: fetcher })
      const agent = await createHarness({ ...options, model })
      let failure: unknown
      try { await agent.invoke(privatePrompt) } catch (error) { failure = error }
      expect(failure).toBeInstanceOf(ModelError)
      if (partial) expect(failure).not.toBeInstanceOf(ModelNetworkError)
      else expect(failure).toBeInstanceOf(ModelNetworkError)
      expectSafe(failure)
      expect((failure as Error).cause).not.toBe(original)
      expect(((failure as Error).cause as Error).cause).toBeUndefined()
      expect(fetcher).toHaveBeenCalledOnce()
    })
  }

  it.each(types)('safely classifies a %s non-streaming body read error', async (ErrorType) => {
    const original = originalError(ErrorType)
    const fetcher = vi.fn(async () => new Response(broken(original)))
    const model = new ExaoneModel({ apiKey: key, stream: false, fetch: fetcher })
    const agent = await createHarness({ ...options, model })
    let failure: unknown
    try { await agent.invoke(privatePrompt) } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelNetworkError)
    expectSafe(failure)
    expect((failure as Error).cause).not.toBe(original)
    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('keeps malformed JSON distinct from a body read failure', async () => {
    const model = new ExaoneModel({ apiKey: key, stream: false, fetch: async () => new Response('{invalid') })
    const agent = await createHarness({ ...options, model })
    let failure: unknown
    try { await agent.invoke('hello') } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelError)
    expect(failure).not.toBeInstanceOf(ModelNetworkError)
    expect((failure as Error).message).toContain('malformed response JSON')
  })

  it('preserves explicit cancellation while an HTTP error body is being read', async () => {
    let reading!: () => void
    const started = new Promise<void>((resolve) => { reading = resolve })
    const fetcher: typeof fetch = async (_url, init) => new Response(new ReadableStream<Uint8Array>({
      start(controller) { init?.signal?.addEventListener('abort', () => controller.error(originalError(Error)), { once: true }) },
      pull() { reading() },
    }), { status: 500 })
    const model = new ExaoneModel({ apiKey: key, fetch: fetcher })
    const controller = new AbortController()
    const message = Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
    const pending = model.stream([message], { cancelSignal: controller.signal }).next()
    await started
    const reason = new Error('cancelled by caller')
    controller.abort(reason)
    await expect(pending).rejects.toBe(reason)
  })
})
