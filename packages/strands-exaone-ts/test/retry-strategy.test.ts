import { createHarness } from '@strands-agents/harness'
import { tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { describe, expect, it, vi } from 'vitest'
import { ExaoneModel, ExaoneRetryStrategy } from '../src/index.js'

const harnessOptions = {
  builtinTools: [] as const,
  builtinPlugins: [] as const,
  session: false as const,
  memory: false as const,
  skills: false as const,
  caching: false as const,
  printer: false as const,
}

const success = () => new Response([
  'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}',
  '',
  'data: [DONE]',
  '',
  '',
].join('\n'), { headers: { 'content-type': 'text/event-stream' } })

describe('EXAONE retry strategy through Strands Harness', () => {
  it('waits for the longest Retry-After hint before retrying a throttled request', async () => {
    let attempts = 0
    const requestTimes: number[] = []
    const fetcher: typeof fetch = async () => {
      attempts += 1
      requestTimes.push(performance.now())
      if (attempts === 1) {
        return new Response(JSON.stringify({ error: { type: 'rate_limit_error', message: 'busy' } }), {
          status: 429,
          headers: { 'retry-after': '1', 'retry-after-ms': '631' },
        })
      }
      return success()
    }
    const agent = await createHarness({
      ...harnessOptions,
      model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }),
      retryStrategy: new ExaoneRetryStrategy({ maxAttempts: 2, baseRetryDelayMs: 1, maxRetryDelayMs: 1_500 }),
    })

    await agent.invoke('Say ok.')

    expect(attempts).toBe(2)
    expect(requestTimes[1] - requestTimes[0]).toBeGreaterThanOrEqual(950)
  })

  it('surfaces a Retry-After hint above the configured cap without retrying early', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ error: { message: 'busy' } }), {
      status: 429,
      headers: { 'retry-after-ms': '100' },
    }))
    const agent = await createHarness({
      ...harnessOptions,
      model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }),
      retryStrategy: new ExaoneRetryStrategy({ maxAttempts: 2, baseRetryDelayMs: 1, maxRetryDelayMs: 50 }),
    })

    await expect(agent.invoke('Say ok.')).rejects.toThrow()

    expect(fetcher).toHaveBeenCalledOnce()
  })

  it('does not retry or execute a tool when the model returns malformed tool arguments', async () => {
    let attempts = 0
    const toolCalls = vi.fn(() => 'should not run')
    const lookup = tool({
      name: 'lookup',
      description: 'Look up a fixed value.',
      inputSchema: z.object({ query: z.string() }),
      callback: toolCalls,
    })
    const malformed = [
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{"}}]},"finish_reason":null}]}',
      '',
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
      '',
      'data: [DONE]',
      '',
      '',
    ].join('\n')
    const fetcher: typeof fetch = async () => {
      attempts += 1
      return new Response(malformed, { headers: { 'content-type': 'text/event-stream' } })
    }
    const agent = await createHarness({
      ...harnessOptions,
      model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }),
      tools: [lookup],
      retryStrategy: new ExaoneRetryStrategy({ maxAttempts: 3, baseRetryDelayMs: 0, maxRetryDelayMs: 0 }),
    })

    await expect(agent.invoke('Use the tool.')).rejects.toThrow()

    expect(attempts).toBe(1)
    expect(toolCalls).not.toHaveBeenCalled()
  })

  it('does not retry malformed model reasoning or expose its raw payload in the error', async () => {
    let attempts = 0
    const fetcher: typeof fetch = async () => {
      attempts += 1
      return new Response([
        'data: {"choices":[{"index":0,"delta":{"reasoning_content":{"invalid":"private payload"}},"finish_reason":null}]}',
        '',
        'data: [DONE]',
        '',
        '',
      ].join('\n'), { headers: { 'content-type': 'text/event-stream' } })
    }
    const agent = await createHarness({
      ...harnessOptions,
      model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }),
      retryStrategy: new ExaoneRetryStrategy({ maxAttempts: 3, baseRetryDelayMs: 0, maxRetryDelayMs: 0 }),
    })

    let failure: unknown
    try {
      await agent.invoke('Continue thinking.')
    } catch (error) {
      failure = error
    }

    expect(failure).toBeInstanceOf(Error)
    expect((failure as Error).message).toContain('invalid reasoning content')
    expect((failure as Error).message).not.toContain('private payload')
    expect(attempts).toBe(1)
  })
})
