import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { ModelError, StructuredOutputError, tool } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const call = (name: string, id: string, args: unknown) => chunk({ tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: JSON.stringify(args) } }] })
const base = { builtinTools: [] as const, builtinPlugins: [] as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }

/** Synthetic transport only; no EXAONE requests are made. */
describe('advanced Strands harness integration', () => {
  it('remains usable after a partial stream failure without executing or replaying its buffered tool', async () => {
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          const pieces = [chunk({ reasoning_content: 'partial reasoning' }), call('lookup', 'unfinished-1', {})]
          controller.enqueue(new TextEncoder().encode(pieces.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('')))
        },
        pull(controller) { controller.error(new TypeError('synthetic stream failure')) },
      }))
      if (requests.length === 2) return response([chunk({ reasoning_content: 'new reasoning' }), chunk({ content: 'Recovered.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...base, model, session: false, tools: [lookup] })
    await expect(agent.invoke('First request')).rejects.toBeInstanceOf(ModelError)
    expect(callback).not.toHaveBeenCalled()
    const result = await agent.invoke('Try again')
    expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'Recovered.' }))
    expect(requests).toHaveLength(2)
    const history = requests[1]!.messages as Record<string, unknown>[]
    expect(JSON.stringify(history)).not.toContain('unfinished-1')
    expect(JSON.stringify(history)).not.toContain('partial reasoning')
    expect(history.at(-1)).toMatchObject({ role: 'user', content: 'Try again' })
    expect(callback).not.toHaveBeenCalled()
  })

  it.each([true, false])('resumes a pending approval after reopening a file-backed session with response %s', async (approved) => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-approval-'))
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => 'approved result')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'approval reasoning' }), call('lookup', 'approval-1', {}), chunk({}, 'tool_calls')])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const options = { ...base, model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), tools: [lookup], interventions: 'ask' as const, session: { id: 'approval', dir } }
    try {
      const first = await createHarness(options)
      const pending = await first.invoke('Use the tool')
      expect(pending.stopReason).toBe('interrupt')
      expect(callback).not.toHaveBeenCalled()
      const interruptId = pending.interrupts![0]!.id
      await first.shutdown()
      const resumed = await createHarness(options)
      await resumed.invoke([{ interruptResponse: { interruptId, response: approved } }])
      await resumed.shutdown()
      expect(callback).toHaveBeenCalledTimes(approved ? 1 : 0)
      expect(requests).toHaveLength(2)
      const history = requests[1]!.messages as Record<string, unknown>[]
      expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'approval reasoning' }))
      expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'approval-1', content: expect.stringContaining(approved ? 'approved result' : 'CONFIRMATION_FAILED') }))
    } finally { await rm(dir, { recursive: true, force: true }) }
  })

  it('passes agentic context controls and tool results through the provider', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'pin reasoning' }), call('pin_context', 'pin-1', { select: 'last_turn', action: 'pin' }), chunk({}, 'tool_calls')])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...base, model, session: false, contextManager: 'agentic' })
    await agent.invoke('Keep this synthetic constraint')
    const toolNames = JSON.stringify(requests[0]!.tools)
    expect(toolNames).toContain('pin_context')
    expect(toolNames).toContain('summarize_context')
    expect(toolNames).toContain('truncate_context')
    expect(JSON.stringify(requests[0]!.messages)).toContain('<context-status>')
    const history = requests[1]!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'pin reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'pin-1', content: expect.stringContaining('Pinned') }))
  })

  it.each([true, false])('delivers a background tool result when success=%s', async (success) => {
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => { if (!success) throw new Error('synthetic background failure'); return 'synthetic background result' })
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'background reasoning' }), call('lookup', 'background-1', { _background_execution: true }), chunk({}, 'tool_calls')])
      if (requests.length <= 4) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...base, model, session: false, tools: [lookup] })
    await agent.invoke('Run in the background')
    expect(callback).toHaveBeenCalledOnce()
    const history = requests.at(-1)!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'background reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'background-1', content: expect.stringContaining('Background task dispatched') }))
    expect(JSON.stringify(history)).toContain(success ? 'synthetic background result' : 'synthetic background failure')
  })

  it('reports invalid forced structured output rather than returning it as valid', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ content: 'Preparing output.' }), chunk({}, 'stop')])
      if (requests.length === 2) return response([chunk({ content: '{"city":17}' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...base, model, session: false, structuredOutputSchema: z.object({ city: z.string() }) })
    await expect(agent.invoke('Return a city')).rejects.toBeInstanceOf(StructuredOutputError)
    expect(requests).toHaveLength(2)
  })
})
