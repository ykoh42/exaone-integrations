import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
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

/** Synthetic data through real file-backed harness features. */
describe('Strands harness skills and memory', () => {
  it('discovers and activates a local skill', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-skills-'))
    const skillDir = join(dir, 'synthetic-skill')
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ reasoning_content: 'synthetic skill reasoning' }),
        chunk({ tool_calls: [{ index: 0, id: 'skill-1', type: 'function', function: { name: 'skills', arguments: JSON.stringify({ skill_name: 'synthetic-skill' }) } }] }),
        chunk({}, 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    try {
      await mkdir(skillDir)
      await writeFile(join(skillDir, 'SKILL.md'), '---\nname: synthetic-skill\ndescription: Synthetic test instructions\n---\n\nUse the synthetic marker sapphire.\n')
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: dir, caching: false, retryStrategy: null, printer: false })
      await agent.invoke('Use the synthetic skill')
      const initial = JSON.stringify(requests[0]!.messages)
      expect(initial).toContain('synthetic-skill')
      expect(initial).toContain('Synthetic test instructions')
      const messages = requests[1]!.messages as Record<string, unknown>[]
      expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'synthetic skill reasoning' }))
      expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'skill-1', content: expect.stringContaining('Use the synthetic marker sapphire.') }))
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('recalls and searches file-backed memory', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-memory-'))
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length === 1) return response([
        chunk({ reasoning_content: 'synthetic memory reasoning' }),
        chunk({ tool_calls: [{ index: 0, id: 'memory-1', type: 'function', function: { name: 'search_memory', arguments: JSON.stringify({ query: 'nebula' }) } }] }),
        chunk({}, 'tool_calls'),
      ])
      if (requests.length === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    try {
      await writeFile(join(dir, 'nebula.md'), '# Nebula project\nThe synthetic color is sapphire.\n')
      const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
      const agent = await createHarness({ model, builtinTools: [], builtinPlugins: [], session: false, memory: { dir }, skills: false, caching: false, retryStrategy: null, printer: false })
      await agent.invoke('What is the nebula project color?')
      expect(JSON.stringify(requests[0]!.messages)).toContain('The synthetic color is sapphire.')
      const messages = requests[1]!.messages as Record<string, unknown>[]
      expect(messages).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'synthetic memory reasoning' }))
      expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'memory-1', content: expect.stringContaining('The synthetic color is sapphire.') }))
      await agent.shutdown()
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})
