import { execFile } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve, sep } from 'node:path'
import { promisify } from 'node:util'
import { createHarness } from '@strands-agents/harness'
import { Sandbox } from '@strands-agents/sdk'
import { describe, expect, it } from 'vitest'
import { ExaoneModel } from '../src/index.js'

const execFileAsync = promisify(execFile)
const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`

/** Synthetic model output driving actual harness file, shell and programmatic tools. */
describe('isolated harness file workflows', () => {
  it('recovers from file errors and edits and searches multiple Unicode paths', async () => {
    const dir = await mkdtemp(join(tmpdir(), "exaone-작업 O'Brien "))
    const first = join(dir, '첫 문서.txt')
    const second = join(dir, '두 문서.txt')
    const missing = join(dir, '새 문서.txt')
    const command = `grep -n '목표' ${quote(first)} ${quote(second)}`
    class FixtureSandbox extends Sandbox {
      private checked(path: string) {
        const candidate = resolve(path)
        if (!candidate.startsWith(dir + sep)) throw new Error('Path outside synthetic workspace')
        return candidate
      }
      async *executeStreaming(value: string) {
        if (value !== command) throw new Error('Unexpected synthetic shell command')
        const { stdout, stderr } = await execFileAsync('/bin/sh', ['-c', value], { cwd: dir })
        yield { type: 'executionResult' as const, exitCode: 0, stdout, stderr }
      }
      executeCodeStreaming(): never { throw new Error('Unexpected sandbox code execution') }
      async readFile(path: string) { return readFile(this.checked(path)) }
      async writeFile(path: string, content: Uint8Array) {
        const target = this.checked(path)
        await mkdir(dirname(target), { recursive: true })
        await writeFile(target, content)
      }
      async removeFile(): Promise<void> { throw new Error('Unexpected sandbox removal') }
      async listFiles(): Promise<[]> { throw new Error('Unexpected sandbox listing') }
    }
    const rounds = [
      [
        { id: 'write-a', name: 'write', args: { path: first, content: '제목\n목표 이전\n' } },
        { id: 'write-b', name: 'write', args: { path: second, content: '목표 둘\n' } },
      ],
      [
        { id: 'read-missing', name: 'read', args: { path: missing } },
        { id: 'edit-missing', name: 'edit', args: { path: first, old_str: '없는 문자열', new_str: '대체' } },
      ],
      [
        { id: 'edit-ok', name: 'edit', args: { path: first, old_str: '목표 이전', new_str: '목표 수정' } },
        { id: 'write-recovery', name: 'write', args: { path: missing, content: '새 파일\n' } },
      ],
      [
        { id: 'search', name: 'shell', args: { command } },
        { id: 'program', name: 'programmatic_tool_caller', args: { code: `print(await read(path=${JSON.stringify(first)}))` } },
      ],
    ]
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      const round = rounds[requests.length - 1]
      if (!round) {
        if (requests.length !== rounds.length + 1) throw new Error('Unexpected synthetic request')
        return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      }
      return response([
        chunk({ reasoning_content: `synthetic reasoning ${requests.length}` }),
        chunk({ tool_calls: round.map((item, index) => ({ index, id: item.id, type: 'function', function: { name: item.name, arguments: JSON.stringify(item.args) } })) }),
        chunk({}, 'tool_calls'),
      ])
    }
    try {
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, sandbox: new FixtureSandbox(), builtinTools: ['read', 'write', 'edit', 'shell', 'programmatic_tool_caller'], builtinPlugins: [], session: { id: 'synthetic', dir }, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
      const result = await agent.invoke('Update and inspect the synthetic files')
      expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'Done.' }))
      expect(await readFile(first, 'utf8')).toBe('제목\n목표 수정\n')
      expect(await readFile(second, 'utf8')).toBe('목표 둘\n')
      expect(await readFile(missing, 'utf8')).toBe('새 파일\n')
      const history = requests.at(-1)!.messages as Record<string, unknown>[]
      const toolResult = (id: string) => history.find((message) => message.role === 'tool' && message.tool_call_id === id)
      expect(String(toolResult('read-missing')?.content)).toContain('ENOENT')
      expect(String(toolResult('edit-missing')?.content)).toContain('did not appear verbatim')
      expect(String(toolResult('search')?.content)).toContain('목표 수정')
      expect(String(toolResult('search')?.content)).toContain('목표 둘')
      expect(String(toolResult('program')?.content)).toContain('목표 수정')
      for (let index = 1; index <= rounds.length; index++) {
        expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: `synthetic reasoning ${index}` }))
      }
      await agent.shutdown()
    } finally { await rm(dir, { recursive: true, force: true }) }
  })

  it('writes a large file and retrieves an offloaded result with the default context manager', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-large-'))
    const path = join(dir, '큰 자료.txt')
    const content = 'synthetic payload line\n'.repeat(5000) + '종료표식\n'
    const prompt = `Inspect a synthetic file: ${'long input '.repeat(5000)}`
    class FixtureSandbox extends Sandbox {
      private checked(path: string) {
        const candidate = resolve(path)
        if (!candidate.startsWith(dir + sep)) throw new Error('Path outside synthetic workspace')
        return candidate
      }
      executeStreaming(): never { throw new Error('Unexpected sandbox command') }
      executeCodeStreaming(): never { throw new Error('Unexpected sandbox code execution') }
      async readFile(path: string) { return readFile(this.checked(path)) }
      async writeFile(path: string, value: Uint8Array) {
        const target = this.checked(path)
        await mkdir(dirname(target), { recursive: true })
        await writeFile(target, value)
      }
      async removeFile(): Promise<void> { throw new Error('Unexpected sandbox removal') }
      async listFiles(): Promise<[]> { throw new Error('Unexpected sandbox listing') }
    }
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      let name: string
      let id: string
      let args: Record<string, unknown>
      if (requests.length === 1) {
        name = 'write'; id = 'large-write'; args = { path, content }
      } else if (requests.length === 2) {
        name = 'read'; id = 'large-read'; args = { path, limit: 6000 }
      } else if (requests.length === 3) {
        const history = requests[2]!.messages as Record<string, unknown>[]
        const result = history.find((item) => item.role === 'tool' && item.tool_call_id === 'large-read')
        const content = String(result?.content)
        const currentReference = content.match(/\[ref: ([^\]]+)\]/)?.[1]
        const reference = currentReference ?? content.match(/\[Stored references:\]\n\s+(\S+)/)?.[1]
        if (!reference) throw new Error('Expected a synthetic offloaded reference')
        name = currentReference ? 'retrieve_context' : 'retrieve_offloaded_content'
        const tools = requests[2]!.tools as { function: { name: string } }[]
        if (!tools.some((tool) => tool.function.name === name)) throw new Error('Expected the matching retrieval tool')
        id = 'large-retrieve'; args = { reference, pattern: '종료표식', context_lines: 0 }
      } else {
        if (requests.length !== 4) throw new Error('Unexpected synthetic request')
        return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      }
      return response([
        chunk({ tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: JSON.stringify(args) } }] }),
        chunk({}, 'tool_calls'),
      ])
    }
    try {
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, sandbox: new FixtureSandbox(), builtinTools: ['read', 'write'], builtinPlugins: [], session: { id: 'large', dir }, memory: false, skills: false, caching: false, retryStrategy: null, printer: false })
      expect(agent.contextManager).toBeDefined()
      await agent.invoke(prompt)
      expect(await readFile(path, 'utf8')).toBe(content)
      expect((requests[0]!.messages as Record<string, unknown>[]).at(-1)).toMatchObject({ role: 'user', content: prompt })
      const history = requests[3]!.messages as Record<string, unknown>[]
      expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'large-retrieve', content: expect.stringContaining('종료표식') }))
      await agent.shutdown()
    } finally { await rm(dir, { recursive: true, force: true }) }
  })
})
