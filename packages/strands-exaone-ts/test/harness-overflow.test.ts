import { createHarness } from '@strands-agents/harness'
import { ContextWindowOverflowError } from '@strands-agents/sdk'
import { describe, expect, it } from 'vitest'
import { ExaoneAPIError, ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }

/** All response errors below are synthetic, not observed EXAONE responses. */
describe('context overflow classification through Strands harness', () => {
  it('retains sanitized API diagnostics as the explicit overflow error cause', async () => {
    const model = new ExaoneModel({ apiKey: 'test-secret', fetch: async () => new Response(JSON.stringify({ error: { code: 'context_length_exceeded', message: 'test-secret' } }), { status: 400 }) })
    const agent = await createHarness({ ...options, model, contextManager: false })
    let failure: unknown
    try { await agent.invoke('synthetic input') } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ContextWindowOverflowError)
    const cause = (failure as Error).cause as ExaoneAPIError
    expect(cause).toBeInstanceOf(ExaoneAPIError)
    expect(cause.status).toBe(400)
    expect(cause.body).toMatchObject({ error: { code: 'context_length_exceeded', message: '[REDACTED]' } })
  })

  const ambiguous = [
    { status: 400, body: { error: { code: null, message: 'A context window setting was invalid' } } },
    { status: 400, body: { error: { code: 'invalid_request_error', message: 'maximum context length setting is invalid' } } },
    { status: 400, body: { error: { code: 'other', message: 'invalid request' }, details: 'context_length_exceeded' } },
    { status: 413, body: { error: { code: null, message: 'request body exceeds a limit' } } },
  ]
  it.each(ambiguous)('does not compact or retry an ambiguous HTTP $status response', async ({ status, body }) => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      if (requests.length <= 3) return response([chunk({ reasoning_content: `reasoning ${requests.length}` }), chunk({ content: `answer ${requests.length}` }), chunk({}, 'stop')])
      return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model })
    for (let i = 1; i <= 3; i++) await agent.invoke(`turn ${i}`)
    const before = JSON.stringify(agent.messages)
    await expect(agent.invoke('failing turn')).rejects.toBeInstanceOf(ExaoneAPIError)
    expect(requests).toHaveLength(4)
    expect(JSON.stringify(agent.messages.slice(0, -1))).toBe(before)
    expect(agent.messages.at(-1)?.content).toContainEqual(expect.objectContaining({ text: 'failing turn' }))
  })

  it.each([400, 413])('allows reactive recovery for an explicit context_length_exceeded code (HTTP %i)', async (status) => {
    const requests: Record<string, unknown>[] = []
    let failed = false
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      const summary = JSON.stringify(request.messages).includes('You are a summarization assistant.')
      if (summary) return response([chunk({ content: 'synthetic summary' }), chunk({}, 'stop')])
      if (requests.length <= 3) return response([chunk({ reasoning_content: `reasoning ${requests.length}` }), chunk({ content: `answer ${requests.length}` }), chunk({}, 'stop')])
      if (!failed) {
        failed = true
        return new Response(JSON.stringify({ error: { code: 'context_length_exceeded', message: 'synthetic error' } }), { status, headers: { 'content-type': 'application/json' } })
      }
      return response([chunk({ content: 'Recovered.' }), chunk({}, 'stop')])
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model })
    for (let i = 1; i <= 3; i++) await agent.invoke(`turn ${i}`)
    const result = await agent.invoke('trigger synthetic overflow')
    expect(result.lastMessage.content).toContainEqual(expect.objectContaining({ text: 'Recovered.' }))
    expect(failed).toBe(true)
    expect(requests.some((request) => JSON.stringify(request.messages).includes('You are a summarization assistant.'))).toBe(true)
    expect(JSON.stringify(requests.at(-1)!.messages)).toContain('synthetic summary')
  })
})
