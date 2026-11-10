import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { describe, expect, it } from 'vitest'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({
  choices: [{ index: 0, delta, finish_reason: finishReason }],
})
const response = (chunks: unknown[]) => new Response(
  chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n',
  { headers: { 'content-type': 'text/event-stream' } },
)

/** Synthetic model responses exercise real harness tools without calling EXAONE. */
describe('Strands harness built-in tools', () => {
  it('uses text descriptions for image reads when media is disabled', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-image-'))
    const path = join(dir, 'sample.png')
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ tool_calls: [{ index: 0, id: 'read-1', type: 'function', function: { name: 'read', arguments: JSON.stringify({ path }) } }] }),
        chunk({}, 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    try {
      await writeFile(path, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/aXcAAAAASUVORK5CYII=', 'base64'))
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, builtinTools: { '*': false, read: { media: false } }, builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
      await agent.invoke('Read the image file')
      const messages = requests[1]!.messages as Record<string, unknown>[]
      expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'read-1', content: expect.stringContaining('[Image file:') }))
      expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', content: expect.stringContaining('contents are not shown') }))
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('writes, reads, edits and searches a temporary file, including a programmatic tool call', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-harness-'))
    const path = join(dir, 'sample.txt')
    const shellPath = `'${path.replaceAll("'", "'\\''")}'`
    const calls = [
      { name: 'write', args: { path, content: 'alpha\nbeta\n' } },
      { name: 'read', args: { path } },
      { name: 'edit', args: { path, old_str: 'beta', new_str: '서울 beta' } },
      { name: 'shell', args: { command: `grep -n beta ${shellPath}` } },
      { name: 'programmatic_tool_caller', args: { code: `print(await read(path=${JSON.stringify(path)}))` } },
    ]
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      const call = calls[requests.length - 1]
      if (call === undefined) {
        if (requests.length !== calls.length + 1) throw new Error('Unexpected synthetic request')
        return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      }
      return response([
        chunk({ reasoning_content: `synthetic reasoning ${requests.length}` }),
        chunk({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }] }),
        chunk({}, 'tool_calls'),
      ])
    }

    try {
      const agent = await createHarness({
        model: new ExaoneModel({ apiKey: 'test-key', fetch: fetcher }),
        builtinTools: ['write', 'read', 'edit', 'shell', 'programmatic_tool_caller'], builtinPlugins: [],
        session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false,
      })
      const result = await agent.invoke('Update and inspect the sample file')
      expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'Done.' }))
      expect(await readFile(path, 'utf8')).toBe('alpha\n서울 beta\n')
      expect(requests).toHaveLength(6)
      const messages = requests.at(-1)!.messages as Record<string, unknown>[]
      for (let index = 1; index <= calls.length; index++) {
        expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: `synthetic reasoning ${index}` }))
        expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: `call_${index}` }))
      }
      const results = messages.filter((message) => message.role === 'tool')
      expect(String(results[1]!.content)).toContain('beta')
      expect(String(results[3]!.content)).toContain('2:서울 beta')
      expect(String(results[4]!.content)).toContain('서울 beta')
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('fetches and summarizes a synthetic page using the configured provider', async () => {
    const requests: Record<string, unknown>[] = []
    let pageRequests = 0
    const server = createServer((_request, reply) => {
      pageRequests++
      reply.writeHead(200, { 'content-type': 'text/html' })
      reply.end('<html><body>The color is blue.</body></html>')
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const address = server.address()
    if (address === null || typeof address === 'string') throw new Error('No local test port')
    const url = `http://127.0.0.1:${address.port}/doc`
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ reasoning_content: 'synthetic fetch reasoning' }),
        chunk({ tool_calls: [{ index: 0, id: 'fetch-1', type: 'function', function: { name: 'web_fetch', arguments: JSON.stringify({ url, prompt: 'Find the color' }) } }] }),
        chunk({}, 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ reasoning_content: 'synthetic summarizer reasoning' }), chunk({ content: 'The color is blue.' }), chunk({}, 'stop')])
      if (requests.length === 3) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    try {
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, builtinTools: { '*': false, web_fetch: { model, transport: 'direct' } }, builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
      await agent.invoke('Fetch the page')
      expect(pageRequests).toBe(1)
      expect(requests).toHaveLength(3)
      expect(JSON.stringify(requests[1]!.messages)).toContain('The color is blue.')
      const messages = requests[2]!.messages as Record<string, unknown>[]
      expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'synthetic fetch reasoning' }))
      expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'fetch-1', content: expect.stringContaining('The color is blue.') }))
      expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', content: expect.stringContaining('synthetic summarizer reasoning') }))
      expect(messages.some((message) => message.reasoning_content === 'synthetic summarizer reasoning')).toBe(false)
    } finally {
      await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
    }
  })

  it('routes a synthetic subagent task through the provider and returns its result', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ reasoning_content: 'synthetic delegation reasoning' }),
        chunk({ tool_calls: [{ index: 0, id: 'delegate-1', type: 'function', function: { name: 'subagent', arguments: JSON.stringify({ task: 'Return the number seven' }) } }] }),
        chunk({}, 'tool_calls'),
      ])
      const messages = requests.at(-1)!.messages as Record<string, unknown>[]
      if (messages.some((message) => message.role === 'user' && message.content === 'Return the number seven')) {
        return response([chunk({ reasoning_content: 'synthetic child reasoning' }), chunk({ content: 'Seven.' }), chunk({}, 'stop')])
      }
      if (requests.length <= 4) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ model, builtinTools: ['subagent'], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
    await agent.invoke('Delegate a small task')
    expect(requests.length).toBeGreaterThanOrEqual(3)
    expect(requests.some((request) => JSON.stringify(request.messages).includes('Return the number seven'))).toBe(true)
    const messages = requests.at(-1)!.messages as Record<string, unknown>[]
    expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'synthetic delegation reasoning' }))
    expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', content: expect.stringContaining('Seven.') }))
    expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', content: expect.stringContaining('synthetic child reasoning') }))
    expect(messages.some((message) => message.reasoning_content === 'synthetic child reasoning')).toBe(false)
  })
})
