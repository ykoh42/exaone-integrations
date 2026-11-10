import { createHarness } from '@strands-agents/harness'
import { AfterToolCallEvent, BeforeToolCallEvent, tool } from '@strands-agents/sdk'
import { HumanInTheLoop } from '@strands-agents/sdk/vended-interventions/hitl'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const response = (chunks: unknown[]) => new Response(chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n')
const call = (name: string, id: string, args: unknown) => chunk({ tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: JSON.stringify(args) } }] })
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, retryStrategy: null, printer: false as const }
const riskRequest = (request: Record<string, unknown>) => JSON.stringify(request.messages).includes('You are a risk evaluator')

/** Synthetic classifications and harmless local tools, with no EXAONE requests. */
describe('Strands harness interventions and hooks', () => {
  it.each([
    { needsApproval: false, approved: false, executes: true },
    { needsApproval: true, approved: true, executes: true },
    { needsApproval: true, approved: false, executes: false },
  ])('uses the smart classifier with requiresApproval=$needsApproval and approval=$approved', async ({ needsApproval, approved, executes }) => {
    const requests: Record<string, unknown>[] = []
    const callback = vi.fn(() => 'synthetic result')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    let parentCalls = 0
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      if (riskRequest(request)) return response([
        chunk({ reasoning_content: 'classifier reasoning' }),
        call('strands_structured_output', 'risk-1', { requiresApproval: needsApproval, reason: 'synthetic decision' }),
        chunk({}, 'tool_calls'),
      ])
      parentCalls++
      if (parentCalls === 1) return response([chunk({ reasoning_content: 'parent reasoning' }), call('lookup', 'parent-1', {}), chunk({}, 'tool_calls')])
      if (parentCalls === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup], interventions: 'smart' })
    const first = await agent.invoke('Use the tool')
    if (needsApproval) {
      expect(first.stopReason).toBe('interrupt')
      expect(callback).not.toHaveBeenCalled()
      expect(first.interrupts).toHaveLength(1)
      await agent.invoke([{ interruptResponse: { interruptId: first.interrupts![0]!.id, response: approved } }])
    } else expect(first.stopReason).toBe('endTurn')
    expect(callback).toHaveBeenCalledTimes(executes ? 1 : 0)
    const classifiers = requests.filter(riskRequest)
    expect(classifiers).toHaveLength(1)
    expect(JSON.stringify(classifiers[0]!.tools)).toContain('strands_structured_output')
    const history = requests.at(-1)!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'parent reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'parent-1', content: expect.stringContaining(executes ? 'synthetic result' : 'CONFIRMATION_FAILED') }))
    expect(JSON.stringify(history)).not.toContain('classifier reasoning')
  })

  it('forces the classifier structured-output tool after an initial plain-text response', async () => {
    const requests: Record<string, unknown>[] = []
    let parentCalls = 0
    let classifierCalls = 0
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      if (riskRequest(request)) {
        classifierCalls++
        if (classifierCalls === 1) return response([chunk({ content: 'Assessing.' }), chunk({}, 'stop')])
        if (classifierCalls === 2) return response([call('strands_structured_output', 'risk-1', { requiresApproval: false, reason: 'synthetic decision' }), chunk({}, 'tool_calls')])
        throw new Error('Unexpected classifier request')
      }
      parentCalls++
      if (parentCalls === 1) return response([call('lookup', 'parent-1', {}), chunk({}, 'tool_calls')])
      if (parentCalls === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const callback = vi.fn(() => 'ok')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup], interventions: 'smart' })
    await agent.invoke('Use the tool')
    const classifiers = requests.filter(riskRequest)
    expect(classifiers).toHaveLength(2)
    expect(classifiers[1]).toMatchObject({ tool_choice: { type: 'function', function: { name: 'strands_structured_output' } } })
    expect(classifiers[1]).not.toHaveProperty('response_format')
    expect(callback).toHaveBeenCalledOnce()
  })

  it('requires approval when the synthetic classifier output fails validation', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      if (riskRequest(request)) {
        const count = requests.filter(riskRequest).length
        return response([chunk({ content: count === 1 ? 'Assessing.' : '{"requiresApproval":"invalid","reason":"synthetic"}' }), chunk({}, 'stop')])
      }
      if (requests.filter((item) => !riskRequest(item)).length === 1) return response([call('lookup', 'parent-1', {}), chunk({}, 'tool_calls')])
      return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
    }
    const callback = vi.fn(() => 'ok')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, tools: [lookup], interventions: 'smart' })
    const pending = await agent.invoke('Use the tool')
    expect(pending.stopReason).toBe('interrupt')
    expect(callback).not.toHaveBeenCalled()
    await agent.invoke([{ interruptResponse: { interruptId: pending.interrupts![0]!.id, response: false } }])
    expect(callback).not.toHaveBeenCalled()
    expect(requests.filter(riskRequest)).toHaveLength(2)
  })

  it.each(['allow', 'deny', 'prompt'] as const)('gates a programmatic nested call with %s', async (decision) => {
    const requests: Record<string, unknown>[] = []
    const before: { name: string; id: string }[] = []
    const after: { name: string; id: string; status: string }[] = []
    const callback = vi.fn(({ value }: { value: string }) => `result: ${value}`)
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({ value: z.string() }), callback })
    const code = 'try:\n    print(await lookup(value="original"))\nexcept RuntimeError:\n    print("guarded")'
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body)) as Record<string, unknown>
      requests.push(request)
      if (riskRequest(request)) return response([call('strands_structured_output', 'risk-1', { requiresApproval: true, reason: 'synthetic decision' }), chunk({}, 'tool_calls')])
      const parentCalls = requests.filter((item) => !riskRequest(item)).length
      if (parentCalls === 1) return response([chunk({ reasoning_content: 'parent reasoning' }), call('programmatic_tool_caller', 'program-1', { code }), chunk({}, 'tool_calls')])
      if (parentCalls === 2) return response([chunk({ content: 'Done.' }), chunk({}, 'stop')])
      throw new Error('Unexpected synthetic request')
    }
    const ask = vi.fn(async () => decision === 'allow')
    const intervention = new HumanInTheLoop({ allowedTools: ['programmatic_tool_caller'], classifier: true, ...(decision === 'prompt' ? {} : { ask }) })
    const model = new ExaoneModel({ apiKey: 'test-key', fetch: fetcher })
    const agent = await createHarness({ ...options, model, builtinTools: ['programmatic_tool_caller'], tools: [lookup], interventions: intervention })
    agent.addHook(BeforeToolCallEvent, (event) => {
      before.push({ name: event.toolUse.name, id: event.toolUse.toolUseId })
      if (event.toolUse.name === 'lookup') event.toolUse.input.value = 'hooked'
    })
    agent.addHook(AfterToolCallEvent, (event) => { after.push({ name: event.toolUse.name, id: event.toolUse.toolUseId, status: event.result.status }) })
    const result = await agent.invoke('Run a synthetic program')
    expect(result.stopReason).toBe('endTurn')
    expect(callback).toHaveBeenCalledTimes(decision === 'allow' ? 1 : 0)
    if (decision === 'allow') expect(callback.mock.calls[0]![0]).toEqual({ value: 'hooked' })
    expect(ask).toHaveBeenCalledTimes(decision === 'prompt' ? 0 : 1)
    const classifiers = requests.filter(riskRequest)
    expect(classifiers).toHaveLength(1)
    expect(JSON.stringify(classifiers[0]!.messages)).toContain('hooked')
    const outer = before.find((item) => item.name === 'programmatic_tool_caller')
    const inner = before.find((item) => item.name === 'lookup')
    expect(outer?.id).toBe('program-1')
    expect(inner?.id).toMatch(/^ptc_/)
    expect(after).toContainEqual(expect.objectContaining({ name: 'lookup', id: inner!.id, status: decision === 'allow' ? 'success' : 'error' }))
    const history = requests.at(-1)!.messages as Record<string, unknown>[]
    expect(history).toContainEqual(expect.objectContaining({ role: 'assistant', reasoning_content: 'parent reasoning' }))
    expect(history).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'program-1', content: expect.stringContaining(decision === 'allow' ? 'result: hooked' : 'guarded') }))
    expect(history.some((message) => message.role === 'tool' && message.tool_call_id === inner!.id)).toBe(false)
  })
})
