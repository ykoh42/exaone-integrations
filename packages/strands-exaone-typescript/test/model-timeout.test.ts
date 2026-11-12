import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { Message } from '@strands-agents/sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ExaoneModel } from '../src/index.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const response = () => new Response('data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
async function collect(model: ExaoneModel, signal?: AbortSignal) {
  for await (const _event of model.stream([user()], { cancelSignal: signal })) { /* consume */ }
}
const stalledFetch: typeof fetch = async (_url, init) => new Promise((_resolve, reject) => {
  const signal = init!.signal!
  if (signal.aborted) reject(signal.reason)
  else signal.addEventListener('abort', () => reject(signal.reason), { once: true })
})

afterEach(() => vi.useRealTimers())

describe('request timeout lifetime', () => {
  it.each(['fetch', 'metadata'] as const)('keeps a standalone Node process alive until a stalled %s times out', (stage) => {
    const path = fileURLToPath(new URL('../dist/index.js', import.meta.url))
    const script = `
      import { pathToFileURL } from 'node:url';
      import { Message } from '@strands-agents/sdk';
      const { ExaoneModel } = await import(pathToFileURL(process.argv[1]).href);
      const stalled = (_url, init) => new Promise((_resolve, reject) => {
        const signal = init.signal;
        if (signal.aborted) reject(signal.reason);
        else signal.addEventListener('abort', () => reject(signal.reason), { once: true });
      });
      const ok = async () => new Response('data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\\n\\ndata: [DONE]\\n\\n');
      const model = new ExaoneModel({ apiKey: 'test-key', timeoutMs: 30, fetch: ${stage === 'fetch' ? 'stalled' : 'ok'}, onMetadata: ${stage === 'metadata' ? '() => new Promise(() => {})' : 'undefined'} });
      try { for await (const _event of model.stream([Message.fromMessageData({role:'user',content:[{text:'hi'}]})])) {} }
      catch (error) { process.stdout.write(error.name); }
    `
    expect(execFileSync(process.execPath, ['--input-type=module', '-e', script, path], { encoding: 'utf8', timeout: 3000 })).toBe('TimeoutError')
  })

  it('waits through the full unsigned 32-bit range without an early timeout', async () => {
    vi.useFakeTimers()
    const model = new ExaoneModel({ apiKey: 'key', timeoutMs: 4_294_967_295, fetch: stalledFetch })
    const pending = collect(model)
    const failure = expect(pending).rejects.toMatchObject({ name: 'TimeoutError' })
    await vi.advanceTimersByTimeAsync(0)
    expect(vi.getTimerCount()).toBe(1)
    await vi.advanceTimersByTimeAsync(4_294_967_294)
    expect(vi.getTimerCount()).toBe(1)
    await vi.advanceTimersByTimeAsync(1)
    await failure
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([1.5, -1, 4_294_967_296, Infinity, NaN])('rejects an out-of-range timeout %s', async (timeoutMs) => {
    expect(() => new ExaoneModel({ apiKey: 'key', timeoutMs, fetch: stalledFetch })).toThrow(RangeError)
  })

  it('supports a zero timeout', async () => {
    const model = new ExaoneModel({ apiKey: 'key', timeoutMs: 0, fetch: stalledFetch })
    await expect(collect(model)).rejects.toMatchObject({ name: 'TimeoutError' })
  })

  it('clears its timer if constructing the combined signal fails', async () => {
    vi.useFakeTimers()
    const model = new ExaoneModel({ apiKey: 'key', timeoutMs: 1000, fetch: stalledFetch })
    await expect(collect(model, {} as AbortSignal)).rejects.toBeInstanceOf(TypeError)
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each(['success', 'fetch-error', 'cancel'] as const)('clears its timer after %s', async (mode) => {
    vi.useFakeTimers()
    const controller = new AbortController()
    const fetcher: typeof fetch = mode === 'success' ? async () => response()
      : mode === 'fetch-error' ? async () => { throw new TypeError('synthetic fetch failure') }
        : stalledFetch
    const model = new ExaoneModel({ apiKey: 'key', timeoutMs: 1000, fetch: fetcher })
    const pending = collect(model, controller.signal)
    if (mode === 'success') await pending
    else {
      const failure = expect(pending).rejects.toBeDefined()
      if (mode === 'cancel') controller.abort(new Error('cancelled'))
      await failure
    }
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not create a timer when no timeout is configured', async () => {
    vi.useFakeTimers()
    await collect(new ExaoneModel({ apiKey: 'key', fetch: async () => response() }))
    expect(vi.getTimerCount()).toBe(0)
  })
})
