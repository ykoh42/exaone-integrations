import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { Sandbox, tool } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const call = (name: string, id: string, args: unknown) => chunk({ tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: JSON.stringify(args) } }] })
const harnessOptions = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }
const isSummaryRequest = (request: Record<string, unknown>) => {
  const messages = request.messages as Record<string, unknown>[]
  return [messages[0], messages.at(-1)].some((message) => String(message?.content).toLowerCase().includes('summar'))
}

/** Synthetic model responses and local resources; no EXAONE requests are made. */
describe('Strands harness feature integration', () => {
  it('injects synthetic environment guidance from the configured sandbox', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-environment-'))
    const requests: Record<string, unknown>[] = []
    class FixtureSandbox extends Sandbox {
      async *executeStreaming(command: string) {
        if (command !== 'pwd' && command !== 'uname -s') throw new Error('Unexpected sandbox command')
        yield { type: 'executionResult' as const, exitCode: 0, stdout: command === 'pwd' ? dir : 'SyntheticOS', stderr: '' }
      }
      executeCodeStreaming(): never { throw new Error('Unexpected sandbox code execution') }
      async readFile(path: string) { return readFile(join(dir, path)) }
      async writeFile() { throw new Error('Unexpected sandbox write') }
      async removeFile() { throw new Error('Unexpected sandbox removal') }
      async listFiles(path: string) {
        return (await readdir(join(dir, path), { withFileTypes: true })).map((entry) => ({ name: entry.name, isDir: entry.isDirectory() }))
      }
    }
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
    }
    try {
      await writeFile(join(dir, 'AGENTS.md'), 'Synthetic environment instructions.\n')
      await mkdir(join(dir, 'nested'))
      await writeFile(join(dir, 'nested', 'README.md'), 'Synthetic nested document.\n')
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ ...harnessOptions, model, sandbox: new FixtureSandbox(), builtinPlugins: ['environment'] })
      await agent.invoke('Inspect the environment')
      await agent.invoke('Inspect it again')
      const sent = JSON.stringify(requests[0]!.messages)
      expect(sent).toContain('Synthetic environment instructions.')
      expect(sent).toContain('SyntheticOS')
      expect(sent).toContain('nested/README.md')
      expect(sent).not.toContain('Synthetic nested document.')
      expect(JSON.stringify(requests[1]!.messages).match(/Synthetic environment instructions\./g)).toHaveLength(1)
    } finally { await rm(dir, { recursive: true, force: true }) }
  })

  it('uses automatic context summarization with a synthetic small context window', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      const isSummary = isSummaryRequest(request)
      return response([chunk(isSummary
        ? { content: 'synthetic summary' }
        : { reasoning_content: `synthetic reasoning ${requests.length}`, content: `synthetic answer ${requests.length}` }, 'stop')])
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher, contextWindowLimit: 1800 })
    const agent = await createHarness({ ...harnessOptions, model, systemPrompt: 'Synthetic instructions' })
    expect(agent.contextManager).toBeDefined()
    for (let index = 0; index < 9; index++) await agent.invoke(`turn-${index}: ${'sample '.repeat(100)}`)
    const summaries = requests.filter(isSummaryRequest)
    expect(summaries.length).toBeGreaterThan(0)
    for (const request of summaries) {
      for (const message of request.messages as Record<string, unknown>[]) {
        if (message.reasoning_content) expect(message.role).toBe('assistant')
      }
    }
    expect(JSON.stringify(requests.at(-1)!.messages)).toContain('synthetic summary')
    expect(JSON.stringify(requests.at(-1)!.messages)).toContain('synthetic reasoning')
  })

  it('injects an updated todo list on the next model call and later turn', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'todo reasoning' }), call('todo_write', 'todo-1', { todos: [{ content: 'Inspect sample', activeForm: 'Inspecting sample', status: 'in_progress' }] }), chunk({}, 'tool_calls')])
      if (requests.length <= 3) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...harnessOptions, model, builtinPlugins: ['todos'] })
    await agent.invoke('Track the task')
    await agent.invoke('What is the next step?')
    expect(JSON.stringify(requests[1]!.messages)).toContain('Inspecting sample')
    expect(JSON.stringify(requests[2]!.messages)).toContain('Inspecting sample')
    const history = requests[2]!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'todo reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'todo-1' }))
    expect(history.some((message) => message.role === 'user' && String(message.content).includes('Inspecting sample'))).toBe(true)
  })

  it.each([true, false])('pauses for tool approval and honors response %s', async (approved) => {
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => 'synthetic result')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'approval reasoning' }), call('lookup', 'approval-1', {}), chunk({}, 'tool_calls')])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...harnessOptions, model, tools: [lookup], interventions: 'ask' })
    const pending = await agent.invoke('Use the tool')
    expect(pending.stopReason).toBe('interrupt')
    expect(callback).not.toHaveBeenCalled()
    expect(pending.interrupts).toHaveLength(1)
    const result = await agent.invoke([{ interruptResponse: { interruptId: pending.interrupts![0]!.id, response: approved } }])
    expect(result.stopReason).toBe('endTurn')
    expect(callback).toHaveBeenCalledTimes(approved ? 1 : 0)
    const history = requests[1]!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'approval reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'approval-1' }))
    const toolResult = history.find((message) => message.role === 'tool')!
    expect(String(toolResult.content)).toContain(approved ? 'synthetic result' : 'CONFIRMATION_FAILED')
  })

  it('returns validated structured output through the forced tool path', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ content: 'Preparing output.' }), chunk({}, 'stop')])
      if (requests.length === 2) return response([chunk({ reasoning_content: 'structured reasoning' }), call('strands_structured_output', 'structured-1', { city: '서울', count: 2 }), chunk({}, 'tool_calls')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...harnessOptions, model, structuredOutputSchema: z.object({ city: z.string(), count: z.number() }) })
    const result = await agent.invoke('Return a structured value')
    expect(result.structuredOutput).toEqual({ city: '서울', count: 2 })
    expect(requests[1]).toMatchObject({ tool_choice: { type: 'function', function: { name: 'strands_structured_output' } } })
    expect(requests[1]).not.toHaveProperty('response_format')
    expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ type: 'reasoningBlock', text: 'structured reasoning' }))
  })

  it('discovers and executes a local stdio MCP tool', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-mcp-'))
    const script = join(dir, 'server.mjs')
    const marker = join(dir, 'called.json')
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([chunk({ reasoning_content: 'mcp reasoning' }), call('local_echo', 'mcp-1', { text: '서울' }), chunk({}, 'tool_calls')])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    let agent: Awaited<ReturnType<typeof createHarness>> | undefined
    try {
      await writeFile(script, `import { createInterface } from 'node:readline';\nimport { writeFile } from 'node:fs/promises';\nconst lines = createInterface({ input: process.stdin });\nlines.on('line', async (line) => {\n  const req = JSON.parse(line);\n  if (req.id === undefined) return;\n  let result;\n  if (req.method === 'initialize') result = { protocolVersion: req.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'synthetic-local', version: '1.0.0' } };\n  else if (req.method === 'tools/list') result = { tools: [{ name: 'echo', description: 'Echo text', inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] } }] };\n  else if (req.method === 'tools/call') { await writeFile(process.argv[2], JSON.stringify(req.params)); result = { content: [{ type: 'text', text: 'echo: ' + req.params.arguments.text }] }; }\n  else result = {};\n  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: req.id, result }) + '\\n');\n});\n`)
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      agent = await createHarness({ ...harnessOptions, model, mcpServers: { local: { command: process.execPath, args: [script, marker], continueOnError: false } } })
      await agent.invoke('Use the local echo tool')
      expect(JSON.parse(await readFile(marker, 'utf8'))).toMatchObject({ name: 'echo', arguments: { text: '서울' } })
      expect(JSON.stringify(requests[0]!.tools)).toContain('local_echo')
      const history = requests[1]!.messages as Record<string, unknown>[]
      expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'mcp reasoning' }))
      expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'mcp-1', content: expect.stringContaining('echo: 서울') }))
    } finally {
      await agent?.shutdown()
      await rm(dir, { recursive: true, force: true })
    }
  })
})
