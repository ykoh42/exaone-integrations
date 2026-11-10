import { getEventListeners } from 'node:events'
import { createHarness } from '@strands-agents/harness'
import { Message, ModelError, ModelNetworkError, ModelThrottledError, tool } from '@strands-agents/sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (events: unknown[]) => new Response(events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n')
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, printer: false as const }

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((accept, fail) => { resolve = accept; reject = fail })
  return { promise, resolve, reject }
}

async function collect(model: ExaoneModel, signal?: AbortSignal) {
  const events = []
  for await (const event of model.stream([user()], { cancelSignal: signal })) events.push(event)
  return events
}

afterEach(() => vi.useRealTimers())

describe('metadata callbacks', () => {
  it('awaits asynchronous metadata before running a tool', async () => {
    const started = deferred<void>()
    const gate = deferred<void>()
    const callback = vi.fn(() => 'done')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher = vi.fn(async () => fetcher.mock.calls.length === 1
      ? response([chunk({ content: 'before tool' }), chunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }), chunk({}, 'tool_calls')])
      : response([chunk({ content: 'done' }), chunk({}, 'stop')]))
    let metadataCalls = 0
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher, onMetadata: async () => {
      if (++metadataCalls === 1) { started.resolve(); await gate.promise }
    } })
    const agent = await createHarness({ ...options, model, tools: [lookup], retryStrategy: null })
    const result = agent.invoke('Use the tool')
    await started.promise
    expect(callback).not.toHaveBeenCalled()
    expect(fetcher).toHaveBeenCalledOnce()
    gate.resolve()
    expect((await result).stopReason).toBe('endTurn')
    expect(callback).toHaveBeenCalledOnce()
    expect(fetcher).toHaveBeenCalledTimes(2)
    await agent.shutdown()
  })

  it.each(['sync', 'async'] as const)('preserves the cause of a %s callback failure', async (kind) => {
    const failure = new Error('synthetic callback failure')
    const onMetadata = kind === 'sync' ? () => { throw failure } : async () => { throw failure }
    const model = new ExaoneModel({ apiKey: 'key', fetch: async () => response([chunk({ content: 'text' }), chunk({}, 'stop')]), onMetadata })
    await expect(collect(model)).rejects.toMatchObject({ message: 'EXAONE metadata callback failed', cause: failure })
  })

  it('accepts a callback returning a value and waits for a returned thenable', async () => {
    const received: unknown[] = []
    const simple = new ExaoneModel({ apiKey: 'key', fetch: async () => response([chunk({ content: 'text' }), chunk({}, 'stop')]), onMetadata: metadata => received.push(metadata) })
    await collect(simple)
    expect(received).toHaveLength(1)

    const started = deferred<void>()
    const gate = deferred<void>()
    const model = new ExaoneModel({ apiKey: 'key', fetch: async () => response([chunk({ content: 'text' }), chunk({}, 'stop')]), onMetadata: () => ({
      then: (resolve: (value: number) => void) => { started.resolve(); void gate.promise.then(() => resolve(42)) },
    }) })
    let finished = false
    const result = collect(model).then(events => { finished = true; return events })
    await started.promise
    expect(finished).toBe(false)
    gate.resolve()
    expect((await result).some(event => event.type === 'modelMetadataEvent')).toBe(true)
  })

  it.each([ModelNetworkError, ModelThrottledError])('does not classify a callback %s as a retryable model failure after partial output', async (RetryableError) => {
    const failure = new RetryableError('synthetic callback failure')
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher = vi.fn(async () => response([chunk({ content: 'visible' }), chunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }), chunk({}, 'tool_calls')]))
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher, onMetadata: async () => { throw failure } })
    const agent = await createHarness({ ...options, model, tools: [lookup] })
    const text: string[] = []
    let error: unknown
    try {
      for await (const event of agent.stream('Use the tool')) {
        if (event.type === 'modelStreamUpdateEvent' && event.event.type === 'modelContentBlockDeltaEvent' && event.event.delta.type === 'textDelta') text.push(event.event.delta.text)
      }
    } catch (caught) { error = caught }
    expect(error).toBeInstanceOf(ModelError)
    expect(error).not.toBeInstanceOf(RetryableError)
    expect(error).toMatchObject({ cause: failure })
    expect(text).toEqual(['visible'])
    expect(fetcher).toHaveBeenCalledOnce()
    expect(callback).not.toHaveBeenCalled()
    await agent.shutdown()
  })

  it('aborts a pending callback, removes its listener, and handles a later rejection', async () => {
    const started = deferred<void>()
    const gate = deferred<void>()
    const controller = new AbortController()
    const reason = new Error('cancelled')
    let requestSignal: AbortSignal | undefined
    const model = new ExaoneModel({ apiKey: 'key', fetch: async (_url, init) => {
      requestSignal = init?.signal ?? undefined
      return response([chunk({ content: 'text' }), chunk({}, 'stop')])
    }, onMetadata: () => { started.resolve(); return gate.promise } })
    const result = collect(model, controller.signal)
    const failure = expect(result).rejects.toBe(reason)
    await started.promise
    expect(getEventListeners(requestSignal!, 'abort')).toHaveLength(1)
    controller.abort(reason)
    await failure
    expect(getEventListeners(requestSignal!, 'abort')).toHaveLength(0)
    gate.reject(new Error('late callback rejection'))
    await new Promise(resolve => setImmediate(resolve))
  })

  it('times out while a callback is pending without waiting for the callback', async () => {
    vi.useFakeTimers()
    const gate = deferred<void>()
    const started = deferred<void>()
    const model = new ExaoneModel({ apiKey: 'key', timeoutMs: 20, fetch: async () => response([chunk({ content: 'text' }), chunk({}, 'stop')]), onMetadata: () => { started.resolve(); return gate.promise } })
    const result = collect(model)
    const failure = expect(result).rejects.toMatchObject({ name: 'TimeoutError' })
    await started.promise
    await vi.advanceTimersByTimeAsync(20)
    await failure
    gate.reject(new Error('late callback rejection'))
    await vi.advanceTimersByTimeAsync(0)
  })

  it.each(['cancel', 'timeout'] as const)('does not execute a pending tool when callback waiting ends in %s', async (mode) => {
    if (mode === 'timeout') vi.useFakeTimers()
    const started = deferred<void>()
    const gate = deferred<void>()
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher = vi.fn(async () => response([chunk({ content: 'visible' }), chunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }), chunk({}, 'tool_calls')]))
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher, ...(mode === 'timeout' ? { timeoutMs: 30 } : {}), onMetadata: () => { started.resolve(); return gate.promise } })
    const agent = await createHarness({ ...options, model, tools: [lookup], retryStrategy: null })
    const run = agent.invoke('Use the tool').then(result => result.stopReason, error => (error as Error).name)
    await started.promise
    if (mode === 'cancel') agent.cancel()
    else await vi.advanceTimersByTimeAsync(30)
    expect(await run).toBe(mode === 'cancel' ? 'cancelled' : 'ModelError')
    expect(callback).not.toHaveBeenCalled()
    expect(fetcher).toHaveBeenCalledOnce()
    gate.reject(new Error('late callback rejection'))
    if (mode === 'timeout') await vi.advanceTimersByTimeAsync(0)
    else await new Promise(resolve => setImmediate(resolve))
    await agent.shutdown()
  })

  it.each([
    ['cancel', false], ['cancel', true], ['timeout', false], ['timeout', true],
  ] as const)('does not emit metadata after %s at the final model event (callback=%s)', async (mode, hasCallback) => {
    vi.useFakeTimers()
    const controller = new AbortController()
    const callback = vi.fn()
    const model = new ExaoneModel({ apiKey: 'key', fetch: async () => response([chunk({ content: 'text' }), chunk({}, 'stop')]), ...(hasCallback ? { onMetadata: callback } : {}), ...(mode === 'timeout' ? { timeoutMs: 100 } : {}) })
    const iterator = model.stream([user()], { cancelSignal: controller.signal })
    const events: string[] = []
    await expect((async () => {
      for await (const event of iterator) {
        events.push(event.type)
        if (event.type === 'modelMessageStopEvent') {
          if (mode === 'cancel') controller.abort(new Error('cancelled'))
          else await vi.advanceTimersByTimeAsync(100)
        }
      }
    })()).rejects.toMatchObject(mode === 'cancel' ? { message: 'cancelled' } : { name: 'TimeoutError' })
    expect(callback).not.toHaveBeenCalled()
    expect(events).not.toContain('modelMetadataEvent')
    expect(vi.getTimerCount()).toBe(0)
  })
})
