import { describe, expect, it, vi } from 'vitest'
import { createHarness } from '@strands-agents/harness'
import { tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel, ExaoneResponseError } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({
  choices: [{ index: 0, delta, finish_reason: finishReason }],
})
const sse = (chunks: unknown[], done = true) =>
  chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + (done ? 'data: [DONE]\n\n' : '')
const response = (text: string) => new Response(text, { headers: { 'content-type': 'text/event-stream' } })

/** A short synthetic conversation; no EXAONE requests are made. */
describe('Strands harness integration', () => {
  it('preserves interleaved reasoning, fragmented text and tool history across a later user turn', async () => {
    const requests: Record<string, unknown>[] = []
    const replies = [
      sse([
        chunk({ reasoning_content: 'reasoning one' }),
        chunk({ content: '\n' }), chunk({ content: '\n' }), chunk({ content: 'First ' }), chunk({ content: '\n' }),
        chunk({ tool_calls: [{ index: 0, id: 'call_', type: 'function', function: { name: 'fir', arguments: '{' } }] }),
        chunk({ tool_calls: [{ index: 0, id: 'one', function: { name: 'st', arguments: '}' } }] }),
        chunk({}, 'tool_calls'),
      ]),
      sse([
        chunk({ reasoning_content: 'reasoning two' }),
        chunk({ tool_calls: [{ index: 0, id: 'call_two', type: 'function', function: { name: 'second', arguments: '{}' } }] }),
        chunk({}, 'tool_calls'),
      ]),
      sse([chunk({ reasoning_content: 'reasoning three' }), chunk({ content: '\n\nAnswer \n' }), chunk({}, 'stop')]),
      sse([chunk({ content: 'Follow-up' }), chunk({}, 'stop')]),
    ]
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      const body = replies[requests.length - 1]
      if (body === undefined) throw new Error('Unexpected synthetic request')
      return response(body)
    }
    const firstCallback = vi.fn(() => '7')
    const secondCallback = vi.fn(() => '3')
    const first = tool({ name: 'first', description: 'Return the first test number', inputSchema: z.object({}), callback: firstCallback })
    const second = tool({ name: 'second', description: 'Return the second test number', inputSchema: z.object({}), callback: secondCallback })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({
      model, tools: [first, second], builtinTools: [], builtinPlugins: [], session: false,
      memory: false, skills: false, caching: false, retryStrategy: null, printer: false,
    })
    expect(agent.contextManager).toBeDefined()

    const result = await agent.invoke('Use both tools in order')
    expect(result.lastMessage.content.find((block) => block.type === 'textBlock')).toMatchObject({ text: '\n\nAnswer \n' })
    await agent.invoke('Follow up on the previous answer')

    expect(firstCallback).toHaveBeenCalledOnce()
    expect(secondCallback).toHaveBeenCalledOnce()
    expect(requests).toHaveLength(4)
    for (const request of requests) {
      expect(request).toMatchObject({ temperature: 1, top_p: 0.95, chat_template_kwargs: { preserve_thinking: true }, stream_options: { include_usage: true } })
    }
    const history = requests[3]!.messages as Record<string, unknown>[]
    const firstAssistant = history.findIndex((item) => item.role === 'assistant' && item.reasoning_content === 'reasoning one')
    const firstResult = history.findIndex((item) => item.role === 'tool' && item.tool_call_id === 'call_one')
    const secondAssistant = history.findIndex((item) => item.role === 'assistant' && item.reasoning_content === 'reasoning two')
    const secondResult = history.findIndex((item) => item.role === 'tool' && item.tool_call_id === 'call_two')
    const lastAssistant = history.findIndex((item) => item.role === 'assistant' && item.reasoning_content === 'reasoning three')
    expect(firstAssistant).toBeGreaterThanOrEqual(0)
    expect(firstAssistant).toBeLessThan(firstResult)
    expect(firstResult).toBeLessThan(secondAssistant)
    expect(secondAssistant).toBeLessThan(secondResult)
    expect(secondResult).toBeLessThan(lastAssistant)
    expect(history[firstAssistant]).toMatchObject({ content: '\n\nFirst \n' })
    expect(history[lastAssistant]).toMatchObject({ content: '\n\nAnswer \n' })
    expect(history.at(-1)).toMatchObject({ role: 'user', content: 'Follow up on the previous answer' })
  })

  it.each([
    { config: { temperature: 0, topP: 0, preserveThinking: false }, expected: { temperature: 0, top_p: 0, chat_template_kwargs: { preserve_thinking: false } } },
    { config: { params: { temperature: 0, top_p: 0.4, chat_template_kwargs: { preserve_thinking: false, enable_thinking: false } } }, expected: { temperature: 0, top_p: 0.4, chat_template_kwargs: { preserve_thinking: false, enable_thinking: false } } },
  ])('passes user overrides through the harness without replacing false or zero', async ({ config, expected }) => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return response(sse([chunk({ content: 'ok' }), chunk({}, 'stop')]))
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher, ...config })
    const agent = await createHarness({ model, builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    expect(agent.contextManager).toBeDefined()
    await agent.invoke('Hello')
    expect(requests).toHaveLength(1)
    expect(requests[0]).toMatchObject(expected)
  })

  it.each(['interrupted', 'error'] as const)('never executes a buffered tool after a synthetic %s stream', async (kind) => {
    const toolStart = chunk({ tool_calls: [{ index: 0, id: 'call', type: 'function', function: { name: 'lookup', arguments: '{}' } }] })
    const body = kind === 'interrupted' ? sse([toolStart], false) : sse([toolStart, { error: { code: 'failed' } }])
    const callback = vi.fn(() => '7')
    const lookup = tool({ name: 'lookup', description: 'Look up a test number', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: async () => response(body) })
    const agent = await createHarness({ model, tools: [lookup], builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    expect(agent.contextManager).toBeDefined()
    if (kind === 'error') await expect(agent.invoke('Use the tool')).rejects.toBeInstanceOf(ExaoneResponseError)
    else await expect(agent.invoke('Use the tool')).rejects.toThrow(/before \[DONE\]/)
    expect(callback).not.toHaveBeenCalled()
  })
})
