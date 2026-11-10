import { createHarness } from '@strands-agents/harness'
import { AfterToolCallEvent, TextBlock, ToolResultBlock, tool } from '@strands-agents/sdk'
import { expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (events: unknown[]) => new Response(events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n')

it('preserves success and error status for parallel tool calls with identical results', async () => {
  const requests: Record<string, unknown>[] = []
  const callback = vi.fn(() => 'same result')
  const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
  const fetcher: typeof fetch = async (_url, init) => {
    requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    if (requests.length === 1) {
      const calls = ['failure', 'success'].map((id, index) => ({ index, id, type: 'function', function: { name: 'lookup', arguments: '{}' } }))
      return response([chunk({ reasoning_content: 'synthetic reasoning' }), chunk({ tool_calls: calls }), chunk({}, 'tool_calls')])
    }
    return response([chunk({ content: 'done' }), chunk({}, 'stop')])
  }
  const agent = await createHarness({ model: new ExaoneModel({ apiKey: 'key', fetch: fetcher }), tools: [lookup], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false, retryStrategy: null })
  agent.addHook(AfterToolCallEvent, event => {
    if (event.toolUse.toolUseId === 'failure') event.result = new ToolResultBlock({ toolUseId: 'failure', status: 'error', content: [new TextBlock('same result')] })
  })
  await agent.invoke('Use both tools')
  const messages = requests[1]!.messages as Record<string, unknown>[]
  expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'synthetic reasoning' }))
  expect(messages).toContainEqual({ role: 'tool', tool_call_id: 'failure', content: '[ERROR] same result' })
  expect(messages).toContainEqual({ role: 'tool', tool_call_id: 'success', content: 'same result' })
  expect(callback).toHaveBeenCalledTimes(2)
  await agent.shutdown()
})
