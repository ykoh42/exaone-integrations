import { createHarness } from '@strands-agents/harness'
import { CachePointBlock, ImageBlock, InvokeModelStage, TextBlock, tool } from '@strands-agents/sdk'
import type { ToolChoice } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }

/** SDK and harness behavior with a synthetic provider transport. */
describe('tool choice and content boundaries', () => {
  const choices: { label: string; choice: ToolChoice; expected: unknown; finish?: string }[] = [
    { label: 'auto', choice: { auto: {} }, expected: 'auto' },
    { label: 'any', choice: { any: {} }, expected: 'required' },
    { label: 'named', choice: { tool: { name: 'lookup' } }, expected: { type: 'function', function: { name: 'lookup' } } },
    { label: 'named with vLLM stop', choice: { tool: { name: 'lookup' } }, expected: { type: 'function', function: { name: 'lookup' } }, finish: 'stop' },
  ]
  it.each(choices)('maps the SDK $label tool choice through harness middleware', async ({ choice, expected, finish }) => {
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => 'synthetic result')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ reasoning_content: 'choice reasoning' }),
        chunk({ tool_calls: [{ index: 0, id: 'choice-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }),
        chunk({}, finish ?? 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup] })
    let first = true
    agent.addMiddleware(InvokeModelStage.Input, async (context) => {
      if (!first) return context
      first = false
      return { ...context, toolChoice: choice }
    })
    await agent.invoke('Use the tool')
    expect(requests[0]!.tool_choice).toEqual(expected)
    expect(requests[1]).not.toHaveProperty('tool_choice')
    const history = requests[1]!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'choice reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'choice-1', content: 'synthetic result' }))
    expect(callback).toHaveBeenCalledOnce()
  })

  it('executes a zero-argument tool when EXAONE sends empty arguments', async () => {
    const argumentsText = ''
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => 'synthetic result')
    const noArguments = tool({ name: 'no_arguments', description: 'Return a fixed value', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ tool_calls: [{ index: 0, id: 'empty-args', type: 'function', function: { name: 'no_arguments', arguments: argumentsText } }] }),
        chunk({}, 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [noArguments] })

    await agent.invoke('Use the tool')

    expect(callback).toHaveBeenCalledOnce()
    expect(requests).toHaveLength(2)
    const history = requests[1]!.messages as Record<string, unknown>[]
    const assistant = history.find((message) => message.role === 'assistant')!
    const toolCalls = assistant.tool_calls as { function: { arguments: string } }[]
    expect(JSON.parse(toolCalls[0]!.function.arguments)).toEqual({})
  })

  it('leaves whitespace arguments to the Strands SDK instead of rewriting them', async () => {
    const callback = vi.fn(() => 'synthetic result')
    const noArguments = tool({ name: 'no_arguments', description: 'Return a fixed value', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async () => response([
      chunk({ tool_calls: [{ index: 0, id: 'blank-args', type: 'function', function: { name: 'no_arguments', arguments: ' \t\n' } }] }),
      chunk({}, 'tool_calls'),
    ])
    const agent = await createHarness({ ...options, model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }), tools: [noArguments] })
    await expect(agent.invoke('Use the tool')).rejects.toThrow()
    expect(callback).not.toHaveBeenCalled()
  })

  it('executes a tool when EXAONE omits only the final object brace', async () => {
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(({ query }: { query: string }) => query)
    const lookup = tool({ name: 'lookup', description: 'Look up a query', inputSchema: z.object({ query: z.string() }), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ tool_calls: [{ index: 0, id: 'lookup-1', type: 'function', function: { name: 'lookup', arguments: '{"query":"EXAONE"' } }] }),
        chunk({}, 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup] })

    await agent.invoke('Use the lookup tool.')

    expect(callback).toHaveBeenCalledOnce()
    expect(callback.mock.calls[0]?.[0]).toEqual({ query: 'EXAONE' })
    const history = requests[1]!.messages as Record<string, unknown>[]
    const assistant = history.find((message) => message.role === 'assistant')!
    expect((assistant.tool_calls as { function: { arguments: string } }[])[0]!.function.arguments).toBe('{"query":"EXAONE"}')
  })

  it('joins SDK system text blocks without changing whitespace', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, systemPrompt: [new TextBlock('First\n'), new TextBlock('\nSecond ')] })
    await agent.invoke('Hello')
    const messages = requests[0]!.messages as Record<string, unknown>[]
    expect(messages[0]).toEqual({ role: 'system', content: 'First\n\nSecond ' })
  })

  it('ignores system cache points and keeps the surrounding text', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, systemPrompt: [new TextBlock('Context'), new CachePointBlock({ cacheType: 'default' })] })
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    try {
      await agent.invoke('Hello')
      const messages = requests[0]!.messages as Record<string, unknown>[]
      expect(messages[0]).toEqual({ role: 'system', content: 'Context' })
      expect(warning).toHaveBeenCalledWith('cache points are not supported in EXAONE system prompts, ignoring cache points')
    } finally {
      warning.mockRestore()
    }
  })

  it('forwards user images and text in order through the harness', async () => {
    const requests: Record<string, unknown>[] = []
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback: () => '7' })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ reasoning_content: 'Plan' }),
        chunk({ tool_calls: [{ index: 0, id: 'image-tool-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }] }),
        chunk({}, 'tool_calls'),
      ])
      return response([chunk({ content: 'Done' }, 'stop')])
    }
    const model = new ExaoneModel({ apiKey: 'test-key', baseURL: 'http://localhost:8000/v1', modelId: 'EXAONE-4.5-33B', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup] })
    await agent.invoke([new TextBlock('First'), new ImageBlock({ format: 'png', source: { bytes: Uint8Array.of(1, 2, 3) } }), new TextBlock('Second'), new ImageBlock({ format: 'jpeg', source: { url: 'https://example.test/image.jpg' } })])
    expect(requests[0]!.messages).toEqual(expect.arrayContaining([expect.objectContaining({ role: 'user', content: [
      { type: 'text', text: 'First' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } },
      { type: 'text', text: 'Second' },
      { type: 'image_url', image_url: { url: 'https://example.test/image.jpg' } },
    ] })]))
    expect(requests[1]!.messages).toEqual(expect.arrayContaining([
      expect.objectContaining({ role: 'assistant', reasoning_content: 'Plan' }),
      expect.objectContaining({ role: 'tool', tool_call_id: 'image-tool-1', content: '7' }),
      expect.objectContaining({ role: 'user', content: (requests[0]!.messages as Record<string, unknown>[]).find((m) => m.role === 'user')!.content }),
    ]))
  })

  it('rejects an image tool result before sending the next request', async () => {
    const fetcher = vi.fn(async () => response([
      chunk({ tool_calls: [{ index: 0, id: 'image-1', type: 'function', function: { name: 'image', arguments: '{}' } }] }),
      chunk({}, 'tool_calls'),
    ]))
    const callback = vi.fn(() => new ImageBlock({ format: 'png', source: { bytes: Uint8Array.of(1, 2, 3) } }))
    const imageTool = tool({ name: 'image', description: 'Return an image', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [imageTool] })
    await expect(agent.invoke('Use the tool')).rejects.toThrow('Unsupported EXAONE tool result content: imageBlock')
    expect(callback).toHaveBeenCalledOnce()
    expect(fetcher).toHaveBeenCalledOnce()
  })
})
