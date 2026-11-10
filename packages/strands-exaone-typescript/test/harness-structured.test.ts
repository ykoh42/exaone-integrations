import { createHarness } from '@strands-agents/harness'
import { InvokeModelStage, tool } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const schema = z.object({
  title: z.string().min(1),
  data: z.object({ tags: z.array(z.string()).min(1), score: z.number().min(0).optional() }),
  notes: z.array(z.object({ text: z.string(), active: z.boolean().optional() })),
})
const valid = { title: '서울', data: { tags: ['one', '둘'], score: 0 }, notes: [{ text: 'ready', active: false }, { text: 'optional' }] }
const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const toolCall = (id: string, name: string, args: unknown) => {
  const text = JSON.stringify(args)
  const middle = Math.floor(text.length / 2)
  return [
    chunk({ tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: text.slice(0, middle) } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: text.slice(middle) } }] }),
    chunk({}, 'tool_calls'),
  ]
}
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }

/** Synthetic responses exercise SDK validation; no EXAONE calls are made. */
describe('structured output through Strands harness', () => {
  it('recovers from invalid nested tool output and preserves history on a later user turn', async () => {
    const requests: Record<string, unknown>[] = []
    const source = { text: '한글\n🙂', count: 0, active: false, nested: [null, { values: ['첫째', '둘째'] }] }
    const callback = vi.fn(() => source)
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'lookup reasoning' }), ...toolCall('lookup-1', 'lookup', {})])
      if (requests.length === 2) return response([chunk({ content: 'Preparing structured output.' }), chunk({}, 'stop')])
      if (requests.length === 3) return response([chunk({ reasoning_content: 'invalid reasoning' }), ...toolCall('invalid-1', 'strands_structured_output', { ...valid, data: { tags: [], score: 'bad' } })])
      if (requests.length === 4) return response([chunk({ reasoning_content: 'valid reasoning' }), ...toolCall('valid-1', 'strands_structured_output', valid)])
      if (requests.length === 5) return response([chunk({ reasoning_content: 'follow-up reasoning' }), ...toolCall('followup-1', 'strands_structured_output', { title: '다음', data: { tags: ['ok'] }, notes: [] })])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup], structuredOutputSchema: schema })
    const first = await agent.invoke('Get structured data')
    expect(first.structuredOutput).toEqual(valid)
    expect(callback).toHaveBeenCalledOnce()
    expect(requests[2]).toMatchObject({ tool_choice: { type: 'function', function: { name: 'strands_structured_output' } } })
    expect(requests[3]).toMatchObject({ tool_choice: { type: 'function', function: { name: 'strands_structured_output' } } })
    const retryHistory = requests[3]!.messages as Record<string, unknown>[]
    expect(retryHistory).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'invalid reasoning' }))
    expect(retryHistory).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'invalid-1', content: expect.stringContaining('data') }))
    const followup = await agent.invoke('Return another object')
    expect(followup.structuredOutput).toEqual({ title: '다음', data: { tags: ['ok'] }, notes: [] })
    const history = requests[4]!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'lookup reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'valid reasoning' }))
    const lookupResult = history.find((message) => message.role === 'tool' && message.tool_call_id === 'lookup-1')
    expect(JSON.parse(String(lookupResult?.content))).toEqual(source)
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'valid-1', content: expect.stringContaining('서울') }))
    expect(history.at(-1)).toMatchObject({ role: 'user', content: 'Return another object' })
  })

  it.each([true, false])('passes a nested JSON Schema through SDK middleware and validates forced JSON text (stream=%s)', async (stream) => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length <= 2) {
        const message = { reasoning_content: `synthetic reasoning ${requests.length}`, content: JSON.stringify(valid) }
        return stream
          ? response([chunk({ reasoning_content: message.reasoning_content }), chunk({ content: message.content }), chunk({}, 'stop')])
          : new Response(JSON.stringify({ choices: [{ index: 0, message, finish_reason: 'stop' }] }))
      }
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher, stream })
    const agent = await createHarness({ ...options, model, structuredOutputSchema: schema })
    agent.addMiddleware(InvokeModelStage.Input, async (context) => ({ ...context, structuredOutputSchema: schema }))
    const result = await agent.invoke('Return structured data')
    expect(result.structuredOutput).toEqual(valid)
    expect(requests).toHaveLength(2)
    expect(requests[0]).toMatchObject({ response_format: { type: 'json_schema', json_schema: { name: 'structured_output', strict: true, schema: { type: 'object', properties: { data: { type: 'object', properties: { tags: { type: 'array' } } }, notes: { type: 'array' } } } } } })
    expect(requests[1]).toMatchObject({ tool_choice: { type: 'function', function: { name: 'strands_structured_output' } } })
    expect(requests[1]).not.toHaveProperty('response_format')
  })

  it('honors the SDK structured-output attempt limit after invalid nested values', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ content: 'Preparing output.' }), chunk({}, 'stop')])
      if (requests.length <= 3) return response([chunk({ content: JSON.stringify({ title: 'bad', data: { tags: [] }, notes: [] }) }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, structuredOutputSchema: schema })
    const result = await agent.invoke('Return structured data', { limits: { structuredOutputAttempts: 2 } })
    expect(result.stopReason).toBe('limitStructuredOutputAttempts')
    expect(result.structuredOutput).toBeUndefined()
    expect(requests).toHaveLength(3)
  })
})
