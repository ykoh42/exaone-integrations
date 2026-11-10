import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { DefaultModelRetryStrategy, ModelError, ModelNetworkError, tool } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({
  choices: [{ index: 0, delta, finish_reason: finishReason }],
})
const sse = (chunks: unknown[]) => chunks.map((value) => `data: ${JSON.stringify(value)}\r\n\r\n`).join('') + 'data: [DONE]\r\n\r\n'
const streamed = (data: string, splitBytes = false) => {
  const bytes = new TextEncoder().encode(data)
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      if (splitBytes) for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
      else controller.enqueue(bytes)
      controller.close()
    },
  }), { headers: { 'content-type': 'text/event-stream' } })
}

/** Synthetic wire data through a real SDK agent and harness. */
describe('Strands harness transport and persistence', () => {
  it.each(['pending', 'reject', 'throw'] as const)('finishes after DONE when transport cancellation is %s', async (mode) => {
    const cancelled = vi.fn(() => {
      if (mode === 'throw') throw new Error('synthetic cleanup failure')
      if (mode === 'reject') return Promise.reject(new Error('synthetic cleanup failure'))
      return new Promise<void>(() => {})
    })
    let body: ReadableStream<Uint8Array> | undefined
    let signal: AbortSignal | null | undefined
    const fetcher: typeof fetch = async (_url, init) => {
      signal = init?.signal
      body = new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(new TextEncoder().encode(sse([chunk({ content: 'complete' }, 'stop')]))) },
        cancel: cancelled,
      })
      return new Response(body)
    }
    const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      const timeout = new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('synthetic cancellation did not finish')), 1000) })
      const result = await Promise.race([agent.invoke('Hello'), timeout])
      expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'complete' }))
      expect(cancelled).toHaveBeenCalledOnce()
      expect(body?.locked).toBe(false)
      expect(signal?.aborted).toBe(true)
    } finally { clearTimeout(timer) }
  })

  it('allows harness cancellation while transport cleanup is pending', async () => {
    let body: ReadableStream<Uint8Array> | undefined
    let agent: Awaited<ReturnType<typeof createHarness>>
    const cancelled = vi.fn(() => { agent.cancel(); return new Promise<void>(() => {}) })
    const fetcher: typeof fetch = async () => {
      body = new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(new TextEncoder().encode(sse([chunk({ content: 'partial' }, 'stop')]))) },
        cancel: cancelled,
      })
      return new Response(body)
    }
    agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      const timeout = new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('synthetic cancellation did not finish')), 1000) })
      const result = await Promise.race([agent.invoke('Hello'), timeout])
      expect(result.stopReason).toBe('cancelled')
      expect(cancelled).toHaveBeenCalledOnce()
      expect(body?.locked).toBe(false)
    } finally { clearTimeout(timer) }
  })

  it('does not feed malformed token counts into harness metrics', async () => {
    let requests = 0
    const onMetadata = vi.fn()
    const fetcher: typeof fetch = async () => {
      requests++
      const usage = requests === 1
        ? { prompt_tokens: -1, completion_tokens: 2, total_tokens: 1 }
        : { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 }
      return streamed(sse([chunk({ content: 'ok' }, 'stop'), { choices: [], usage }]))
    }
    const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher, onMetadata }), builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    const first = await agent.invoke('First turn')
    expect(first.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'ok' }))
    expect(first.metrics?.latestContextSize).toBeUndefined()
    expect(onMetadata.mock.calls[0]![0].usage?.prompt_tokens).toBe(-1)
    const second = await agent.invoke('Second turn')
    expect(second.metrics?.latestContextSize).toBe(4)
    expect(second.metrics?.projectedContextSize).toBe(6)
  })

  it.each(['content', 'reasoning_content'] as const)('does not replay partial %s on a network failure with the SDK default retry policy', async (kind) => {
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    let requests = 0
    const fetcher: typeof fetch = async () => {
      requests++
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          const payload = sse([
            chunk({ tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }),
            chunk({ [kind]: 'partial' }),
          ]).replace('data: [DONE]\r\n\r\n', '')
          controller.enqueue(new TextEncoder().encode(payload))
        },
        pull(controller) { controller.error(new TypeError('synthetic failure contains test-secret')) },
      }))
    }
    const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-secret', fetch: fetcher }), tools: [lookup], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false })
    const fragments: string[] = []
    let failure: unknown
    try {
      for await (const event of agent.stream('Look up the value')) {
        if (event.type === 'modelStreamUpdateEvent' && event.event.type === 'modelContentBlockDeltaEvent' && 'text' in event.event.delta) fragments.push(event.event.delta.text)
      }
    } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelError)
    expect(failure).not.toBeInstanceOf(ModelNetworkError)
    expect((failure as Error).message).toContain('after content was delivered')
    expect(((failure as Error).cause as Error).message).toContain('[REDACTED]')
    expect(((failure as Error).cause as Error).message).not.toContain('test-secret')
    expect(fragments).toEqual(['partial'])
    expect(requests).toBe(1)
    expect(callback).not.toHaveBeenCalled()
  })

  it.each(['request', 'stream'] as const)('lets the SDK retry a %s network failure before response content', async (failure) => {
    let requests = 0
    const fetcher: typeof fetch = async () => {
      requests++
      if (requests === 1) {
        if (failure === 'request') throw new TypeError('synthetic connection failure')
        return new Response(new ReadableStream<Uint8Array>({ start(controller) { controller.error(new TypeError('synthetic stream failure')) } }))
      }
      return streamed(sse([chunk({ content: 'ok' }), chunk({}, 'stop')]))
    }
    const retryStrategy = new DefaultModelRetryStrategy({ maxAttempts: 2, backoff: { nextDelay: () => 0 } })
    const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy, printer: false })
    const result = await agent.invoke('Hello')
    expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'ok' }))
    expect(requests).toBe(2)
  })

  it('cancels a partial stream without executing the buffered tool', async () => {
    const aborted = vi.fn()
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    let requests = 0
    const fetcher: typeof fetch = async (_url, init) => {
      requests++
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          init?.signal?.addEventListener('abort', () => { aborted(); controller.error(init.signal?.reason) }, { once: true })
          controller.enqueue(new TextEncoder().encode(sse([
            chunk({ tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }),
            chunk({ reasoning_content: 'partial' }),
          ]).replace('data: [DONE]\r\n\r\n', '')))
        },
      }))
    }
    const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), tools: [lookup], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    let sawPartial = false
    try {
      for await (const event of agent.stream('Look up the value')) {
        if (event.type === 'modelStreamUpdateEvent' && event.event.type === 'modelContentBlockDeltaEvent' && event.event.delta.type === 'reasoningContentDelta') {
          sawPartial = true
          agent.cancel()
        }
      }
    } catch (error) {
      expect(error).toBeInstanceOf(Error)
    }
    expect(sawPartial).toBe(true)
    expect(requests).toBe(1)
    expect(callback).not.toHaveBeenCalled()
    expect(aborted).toHaveBeenCalledOnce()
  })

  it('reassembles byte-split UTF-8 and interleaved parallel tool calls', async () => {
    const requests: Record<string, unknown>[] = []
    const calls = [
      chunk({ reasoning_content: '분석' }), chunk({ content: '\n\n준비 \n' }),
      chunk({ tool_calls: [{ index: 0, id: 'call_', type: 'function', function: { name: 'fir', arguments: '{"city":"' } }] }),
      chunk({ tool_calls: [{ index: 1, id: 'call_', type: 'function', function: { name: 'sec', arguments: '{"city":"' } }] }),
      chunk({ tool_calls: [{ index: 1, id: 'two', function: { name: 'ond', arguments: '부산"}' } }] }),
      chunk({ tool_calls: [{ index: 0, id: 'one', function: { name: 'st', arguments: '서울"}' } }] }),
      chunk({}, 'tool_calls'),
    ]
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return streamed(sse(calls), true)
      if (requests.length === 2) return streamed(sse([chunk({ content: '완료' }), chunk({}, 'stop')]), true)
      throw new Error('Unexpected synthetic request')
    }
    const firstCallback = vi.fn(({ city }: { city: string }) => `first: ${city}`)
    const secondCallback = vi.fn(({ city }: { city: string }) => `second: ${city}`)
    const first = tool({ name: 'first', description: 'First lookup', inputSchema: z.object({ city: z.string() }), callback: firstCallback })
    const second = tool({ name: 'second', description: 'Second lookup', inputSchema: z.object({ city: z.string() }), callback: secondCallback })
    const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), tools: [first, second], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    const result = await agent.invoke('Look up both cities')
    expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: '완료' }))
    expect(firstCallback).toHaveBeenCalledOnce()
    expect(firstCallback.mock.calls[0]![0]).toEqual({ city: '서울' })
    expect(secondCallback).toHaveBeenCalledOnce()
    expect(secondCallback.mock.calls[0]![0]).toEqual({ city: '부산' })
    const messages = requests[1]!.messages as Record<string, unknown>[]
    expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: '분석', content: '\n\n준비 \n' }))
    expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'call_one', content: expect.stringContaining('서울') }))
    expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'call_two', content: expect.stringContaining('부산') }))
  })

  it('restores reasoning and the prior answer when resuming a saved session', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-session-'))
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return streamed(sse([chunk({ reasoning_content: 'prior reasoning' }), chunk({ content: 'prior answer' }), chunk({}, 'stop')]))
      if (requests.length === 2) return streamed(sse([chunk({ content: 'continued' }), chunk({}, 'stop')]))
      throw new Error('Unexpected synthetic request')
    }
    const options = { model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), builtinTools: [] as const, builtinPlugins: [] as const, session: { id: 'resume-test', dir }, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }
    try {
      const first = await createHarness(options)
      await first.invoke('First turn')
      await first.shutdown()
      const resumed = await createHarness(options)
      await resumed.invoke('Next turn')
      await resumed.shutdown()
      expect(requests).toHaveLength(2)
      const messages = requests[1]!.messages as Record<string, unknown>[]
      expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'prior reasoning', content: 'prior answer' }))
      expect(messages.at(-1)).toMatchObject({ role: 'user', content: 'Next turn' })
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})
