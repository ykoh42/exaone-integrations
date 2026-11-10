import { createHarness } from '@strands-agents/harness'
import { ModelError, ModelNetworkError, tool } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'
import { ResponseMetadataCollector } from '../src/metadata.js'

const event = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (payloads: unknown[]) => new Response(
  payloads.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n',
  { headers: { 'content-type': 'text/event-stream' } },
)

/** Synthetic malformed payloads; these are not EXAONE service observations. */
describe('malformed response handling through Strands harness', () => {
  const cases: { name: string; payload: (field: 'delta' | 'message') => unknown }[] = [
    { name: 'null response', payload: () => null },
    { name: 'array response', payload: () => [] },
    { name: 'object choices', payload: () => ({ choices: {} }) },
    { name: 'null choice', payload: () => ({ choices: [null] }) },
    { name: 'array choice', payload: () => ({ choices: [[]] }) },
    { name: 'array message', payload: (field) => ({ choices: [{ [field]: [] }] }) },
    { name: 'object tool calls', payload: (field) => ({ choices: [{ [field]: { tool_calls: {} } }] }) },
    { name: 'null tool call', payload: (field) => ({ choices: [{ [field]: { tool_calls: [null] } }] }) },
    { name: 'string tool call', payload: (field) => ({ choices: [{ [field]: { tool_calls: ['invalid'] } }] }) },
    { name: 'array function', payload: (field) => ({ choices: [{ [field]: { tool_calls: [{ index: 0, function: [] }] } }] }) },
  ]
  for (const stream of [true, false]) {
    it.each(cases)(`rejects $name without retry or tool execution (stream=${stream})`, async ({ payload }) => {
      const callback = vi.fn(() => 'unexpected')
      const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
      const fetcher = vi.fn(async () => stream
        ? response([payload('delta')])
        : new Response(JSON.stringify(payload('message')), { headers: { 'content-type': 'application/json' } }))
      const model = new ExaoneModel({ apiKey: 'test-key', stream, fetch: fetcher })
      const agent = await createHarness({ model, tools: [lookup], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false })
      let failure: unknown
      try { await agent.invoke('Use the tool') } catch (error) { failure = error }
      expect(failure).toBeInstanceOf(ModelError)
      expect(failure).not.toBeInstanceOf(ModelNetworkError)
      expect(fetcher).toHaveBeenCalledOnce()
      expect(callback).not.toHaveBeenCalled()
    })
  }

  it('does not misclassify an internal mapping TypeError as a network failure', async () => {
    const mapping = vi.spyOn(ResponseMetadataCollector.prototype, 'accept').mockImplementation(() => { throw new TypeError('synthetic internal failure') })
    const fetcher = vi.fn(async () => response([event({ content: 'ok' }, 'stop')]))
    try {
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false })
      await expect(agent.invoke('Hello')).rejects.toThrow('synthetic internal failure')
      expect(fetcher).toHaveBeenCalledOnce()
    } finally { mapping.mockRestore() }
  })

  it('rejects conflicting finish reasons before exposing buffered tools', async () => {
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher = vi.fn(async () => response([
      event({ tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }),
      event({}, 'length'), event({}, 'stop'),
    ]))
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ model, tools: [lookup], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false })
    await expect(agent.invoke('Use the tool')).rejects.toThrow('conflicting finish reasons')
    expect(fetcher).toHaveBeenCalledOnce()
    expect(callback).not.toHaveBeenCalled()
  })
})
