import { describe, expect, it, vi } from 'vitest'
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({
  choices: [{ index: 0, delta, finish_reason: finishReason }],
})
const sse = (chunks: unknown[]) => `${chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('')}data: [DONE]\n\n`

describe('assistant text fidelity', () => {
  it.each([true, false])('preserves leading and trailing whitespace across an agent tool turn (stream=%s)', async (stream) => {
    const requests: Record<string, unknown>[] = []
    const firstText = '\n\nFirst answer \n'
    const finalText = '\n\nFinal answer \n'
    const firstCall = { id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }
    const first = stream
      ? sse([
        chunk({ reasoning_content: 'need lookup' }),
        chunk({ content: '\n' }), chunk({ content: '\n' }), chunk({ content: 'First answer' }), chunk({ content: ' \n' }),
        chunk({ tool_calls: [{ index: 0, ...firstCall }] }), chunk({}, 'tool_calls'),
      ])
      : JSON.stringify({ choices: [{ index: 0, message: { reasoning_content: 'need lookup', content: firstText, tool_calls: [firstCall] }, finish_reason: 'tool_calls' }] })
    const second = stream
      ? sse([chunk({ content: '\n' }), chunk({ content: '\n' }), chunk({ content: 'Final answer' }), chunk({ content: ' \n' }), chunk({}, 'stop')])
      : JSON.stringify({ choices: [{ index: 0, message: { content: finalText }, finish_reason: 'stop' }] })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return new Response(requests.length === 1 ? first : second)
    }
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up a number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', stream, fetch: fetcher })
    const result = await new Agent({ model, tools: [lookup], printer: false, retryStrategy: null }).invoke('Use the tool')

    expect(callback).toHaveBeenCalledOnce()
    const history = requests[1]!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', content: firstText, reasoning_content: 'need lookup' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'call-1' }))
    expect(result.lastMessage.content.find((block) => block.type === 'textBlock')).toMatchObject({ text: finalText })
  })
})
