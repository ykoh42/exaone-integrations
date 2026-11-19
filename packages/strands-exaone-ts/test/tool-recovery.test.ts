import { describe, expect, it } from 'vitest'
import { inspect } from 'node:util'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Agent, InvokeModelStage, MaxTokensError, Message, ModelStreamUpdateEvent, SessionManager, tool } from '@strands-agents/sdk'
import { LocalFileStorage } from '@strands-agents/sdk/storage'
import { z } from 'zod'
import { ExaoneModel, toolCallRecovery } from '../src/index.js'

type Call = { id: string; type: 'function'; function: { name: string; arguments: string } }
const call = (id: string, args: string): Call => ({ id, type: 'function', function: { name: 'transfer', arguments: args } })
type Case = 'normal' | 'malformed' | 'repeat' | 'mixed' | 'length' | 'secret' | 'cancel' | 'cancel-correction' | 'partial' | 'usage_missing' |
  'fallback' | 'bad-request' | 'no-tool' | 'wrong-tool' | 'extra-tool' | 'blank' | 'array' | 'multiline' | 'tool-only' | 'live-text' |
  'repeated-original-id' | 'repeated-correction-id' | 'refusal-with-tool' | 'filtered-initial' | 'correction-length' |
  'routed-wrong-tool' | 'routed-required-wrong-tool'
const namedChoice = { type: 'function', function: { name: 'transfer' } }
const multilineArguments = '{\n"note":"😀```","amount":123,,}'

class ObservedFileStorage extends LocalFileStorage {
  readonly writes: string[] = []

  override async write(key: string, data: Uint8Array): Promise<void> {
    this.writes.push(new TextDecoder().decode(data))
    await super.write(key, data)
  }
}

async function run(scenario: Case, streaming: boolean, enabled = true, requireTool = false, sessionManager?: SessionManager) {
  const requests: Record<string, any>[] = []
  const executions: number[] = []
  const observed: string[] = []
  const sequence: string[] = []
  const metadata: any[] = []
  const sdkUsage: any[] = []
  const controller = new AbortController()
  let finishResponse: (() => void) | undefined
  let exposedBeforeRetry: number | undefined
  const routed = scenario === 'routed-wrong-tool' || scenario === 'routed-required-wrong-tool'
  const fallback = scenario === 'fallback' || scenario === 'routed-required-wrong-tool'
  const fetcher: typeof fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as Record<string, any>
    requests.push(body)
    const normalToolCall = scenario === 'normal' || scenario === 'refusal-with-tool'
    if (requests.length === 2 && !normalToolCall) exposedBeforeRetry = observed.length
    const first = requests.length === 1
    if (requests.length === 2 && (fallback || scenario === 'bad-request')) {
      return Response.json({ error: fallback
        ? { message: 'Named tool choice is not supported' }
        : { message: 'invalid request to model' } }, { status: 400 })
    }
    const correcting = requests.length === (fallback ? 3 : 2) && !normalToolCall
    const bad = call('failed', scenario === 'secret' ? '{"secret":"fixture%2Bkey",}' : scenario === 'partial' ? '{"amount":12' :
      scenario === 'multiline' ? multilineArguments : '{"amount":123,}')
    const good = call(first ? 'original' : 'corrected', '{"amount":123}')
    const replacement = scenario === 'wrong-tool' || routed ? [{ ...good, function: { ...good.function, name: 'other' } }]
      : scenario === 'extra-tool' ? [good, { ...good, id: 'additional' }]
      : scenario === 'no-tool' ? null
      : scenario === 'repeat' ? [bad]
      : scenario === 'blank' || scenario === 'array' ? [call('corrected', scenario === 'blank' ? ' ' : '[]')]
      : [good]
    const calls = first
      ? normalToolCall ? [good] : scenario === 'mixed' ? [bad, good] : [bad]
      : correcting ? replacement : null
    const truncated = (first && scenario === 'length') || (correcting && scenario === 'correction-length')
    const finish = truncated ? 'length' : calls ? 'tool_calls' : 'stop'
    const showReasoning = first && !normalToolCall && !['tool-only', 'repeated-correction-id'].includes(scenario)
    const refusal = first && scenario === 'refusal-with-tool'
    const message = calls
      ? { content: refusal ? 'Synthetic reply.' : correcting ? 'private correction preamble' : null, ...(refusal ? { refusal: 'Declined.' } : {}),
          ...(showReasoning ? { reasoning_content: 'synthetic reasoning' } : {}), tool_calls: calls }
      : { content: 'complete' }
    const usage = first && scenario === 'usage_missing' ? undefined : first
      ? { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12, prompt_tokens_details: { cached_tokens: 2, cache_write_tokens: 1 } }
      : { prompt_tokens: 9, completion_tokens: 6, total_tokens: 15, prompt_tokens_details: { cached_tokens: 3, cache_write_tokens: 0 } }
    if ((first && scenario === 'cancel') || (correcting && scenario === 'cancel-correction')) controller.abort()
    if (!streaming) return new Response(JSON.stringify({ choices: [{ index: 0, message, finish_reason: finish }], usage }))
    const delta = { ...message, ...(calls ? { tool_calls: calls.map((item, index) => ({ ...item, index })) } : {}) }
    if ((first && scenario === 'repeated-original-id') || (correcting && scenario === 'repeated-correction-id')) {
      const frames = [
        { choices: [{ index: 0, delta: { tool_calls: [{ ...calls![0], index: 0 }] } }] },
        { choices: [{ index: 0, delta: { tool_calls: [{ id: calls![0]!.id, index: 0 }] }, finish_reason: finish }], usage },
      ]
      return new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n')
    }
    if (first && scenario === 'live-text') {
      const encoder = new TextEncoder()
      const body = new ReadableStream<Uint8Array>({
        start(stream) {
          stream.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { ...delta, content: 'Checking.' } }] })}\n\n`))
          finishResponse = () => {
            stream.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: finish }], usage })}\n\ndata: [DONE]\n\n`))
            stream.close()
            finishResponse = undefined
          }
        },
        cancel() { finishResponse = undefined },
      })
      return new Response(body)
    }
    return new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }], usage })}\n\ndata: [DONE]\n\n`)
  }
  const record = tool({ name: 'transfer', description: 'Record a synthetic value', inputSchema: z.object({ amount: z.number().int() }),
    callback: ({ amount }) => { executions.push(amount); return 'recorded' } })
  const other = tool({ name: 'other', description: 'Record an unrelated value', inputSchema: z.object({ amount: z.number().int() }),
    callback: ({ amount }) => { executions.push(amount); return 'recorded' } })
  const model = new ExaoneModel({ apiKey: 'fixture+key', fetch: fetcher, stream: streaming, maxRetries: 0, onMetadata: (value) => { metadata.push(value); sequence.push('onMetadata') } })
  const agent = new Agent({ id: 'recovery-agent', model, tools: routed ? [record, other] : [record], systemPrompt: 'Unchanged system instruction', printer: false, retryStrategy: null,
    ...(sessionManager === undefined ? {} : { sessionManager }) })
  agent.addHook(ModelStreamUpdateEvent, (event) => {
    observed.push(event.event.type)
    sequence.push(event.event.type)
    if (event.event.type === 'modelMetadataEvent') sdkUsage.push(event.event.usage)
    if (event.event.type === 'modelContentBlockDeltaEvent' && event.event.delta.type === 'textDelta') finishResponse?.()
  })
  if (requireTool) agent.addMiddleware(InvokeModelStage.Input, (context) => context.messages.length === 1 ? { ...context, toolChoice: { any: {} } } : context)
  if (enabled) agent.addMiddleware(InvokeModelStage, toolCallRecovery())
  if (scenario === 'filtered-initial') {
    agent.addMiddleware(InvokeModelStage, async function* (context, next) {
      return yield* next({ ...context, toolSpecs: [] })
    })
  }
  let failure: unknown
  try { await agent.invoke('Call transfer using amount 123', { cancelSignal: controller.signal }) } catch (error) { failure = error }
  return { requests, executions, observed, exposedBeforeRetry, failure, metadata, sdkUsage, sequence,
    agentMessages: agent.messages.map((message) => message.toJSON()) }
}

describe('toolCallRecovery', () => {
  it.each(['routed-wrong-tool', 'routed-required-wrong-tool'] as const)('validates the final SDK result on %s', async (scenario) => {
    const result = await run(scenario, true)
    const fallback = scenario === 'routed-required-wrong-tool'
    expect(result.failure).toMatchObject({ message: expect.stringContaining('original tool') })
    expect(result.requests.map((request) => request.tool_choice)).toEqual([undefined, namedChoice, ...(fallback ? ['required'] : [])])
    expect(result.executions).toEqual([])
    expect(result.observed).not.toContain('modelMessageStopEvent')
  })

  it('honors tool filtering on the effective model request', async () => {
    const result = await run('filtered-initial', false)
    expect(result.failure).toBeDefined()
    expect(result.requests).toHaveLength(1)
    expect(result.requests[0]?.tools).toBeUndefined()
    expect(result.executions).toEqual([])
  })

  it.each([false, true])('preserves the SDK max-token error on correction (stream=%s)', async (streaming) => {
    const result = await run('correction-length', streaming)
    expect(result.failure).toBeInstanceOf(MaxTokensError)
    expect(result.requests).toHaveLength(2)
    expect(result.executions).toEqual([])
  })

  it.each([false, true])('blocks refused tool calls with text only when recovery is enabled (stream=%s)', async (streaming) => {
    const plain = await run('refusal-with-tool', streaming, false)
    const guarded = await run('refusal-with-tool', streaming, true, false)
    expect(plain.failure).toBeUndefined()
    expect(plain.requests).toHaveLength(2)
    expect(plain.executions).toEqual([123])
    expect(guarded.failure).toMatchObject({ message: expect.stringContaining('refusal text together with tool calls') })
    expect(guarded.requests).toHaveLength(1)
    expect(guarded.executions).toEqual([])
  })

  it.each(['repeated-original-id', 'repeated-correction-id'] as const)('rejects an ambiguous tool call ID (%s)', async (scenario) => {
    const result = await run(scenario, true, true, false)
    expect(result.failure).toMatchObject({ message: expect.stringContaining('ambiguous') })
    expect(result.requests).toHaveLength(scenario === 'repeated-original-id' ? 1 : 2)
    expect(result.executions).toEqual([])
    expect(result.observed).toEqual([])
  })

  it('streams text immediately and completes the same message after correcting the tool', async () => {
    const result = await run('live-text', true)
    expect(result.failure).toBeUndefined()
    expect(result.exposedBeforeRetry).toBeGreaterThan(0)
    expect(result.requests).toHaveLength(3)
    expect(result.executions).toEqual([123])
    expect(result.observed.filter((event) => event === 'modelMessageStartEvent')).toHaveLength(2)
    expect(result.requests[2]?.messages.at(-2)).toMatchObject({ content: 'Checking.' })
  })

  it('recovers a tool-only stream without buffering ordinary text', async () => {
    const result = await run('tool-only', true, true, false)
    expect(result.failure).toBeUndefined()
    expect(result.exposedBeforeRetry).toBe(0)
    expect(result.requests.map((request) => request.tool_choice)).toEqual([undefined, namedChoice, undefined])
    expect(result.executions).toEqual([123])
  })

  it.each([false, true])('keeps original events and corrects only tool arguments (stream=%s)', async (streaming) => {
    const result = await run('malformed', streaming)
    expect(result.failure).toBeUndefined()
    expect(result.requests).toHaveLength(3)
    expect(result.exposedBeforeRetry).toBeGreaterThan(0)
    expect(result.executions).toEqual([123])
    expect(result.metadata[0].recovery.failedAttempts[0].usage).toMatchObject({ prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 })
    expect(result.sdkUsage[0]).toMatchObject({ inputTokens: 16, outputTokens: 11, totalTokens: 27, cacheReadInputTokens: 5, cacheWriteInputTokens: 1 })
    const [first, retry] = result.requests
    expect(retry?.messages.filter((message: any) => message.role === 'user')).toEqual(first?.messages.filter((message: any) => message.role === 'user'))
    expect(retry?.messages.filter((message: any) => message.role === 'system')).toEqual(first?.messages.filter((message: any) => message.role === 'system'))
    expect(first?.tool_choice).toBeUndefined()
    expect(retry?.tool_choice).toEqual(namedChoice)
    expect(result.requests[2]?.tool_choice).toBeUndefined()
    const firstOptions = { ...first }, retryOptions = { ...retry }
    delete firstOptions.messages
    delete retryOptions.messages
    delete retryOptions.tool_choice
    expect(retryOptions).toEqual(firstOptions)
    const prior = retry?.messages.find((message: any) => message.role === 'assistant')
    expect(prior.reasoning_content).toBe('synthetic reasoning')
    expect(prior.tool_calls[0]).toMatchObject({ id: 'failed', function: { name: 'transfer', arguments: '{}' } })
    const feedback = retry?.messages.find((message: any) => message.role === 'tool')
    expect(feedback.tool_call_id).toBe('failed')
    const [header, diagnostic, instruction] = String(feedback.content).split('\n')
    expect(header).toMatch(/^\[ERROR\] Error: .*not executed\. The previous tool call is a placeholder\.$/)
    expect(JSON.parse(diagnostic!)).toMatchObject({ raw_arguments: '{"amount":123,}', parse_error: expect.stringContaining('line 1, column 15') })
    expect(instruction).toMatch(/^Call transfer once/)
    const finalRequest = result.requests[2]?.messages
    const finalCalls = finalRequest.filter((message: any) => message.role === 'assistant').flatMap((message: any) => message.tool_calls ?? [])
    const finalResults = finalRequest.filter((message: any) => message.role === 'tool')
    expect(finalCalls).toEqual([call('failed', '{"amount":123}')])
    expect(finalResults).toHaveLength(1)
    expect(finalResults[0].tool_call_id).toBe('failed')
    expect(JSON.stringify(finalRequest)).toContain('synthetic reasoning')
    expect(JSON.stringify(finalRequest)).not.toContain('{"amount":123,}')
    const serializedHistory = JSON.stringify(result.agentMessages)
    expect(serializedHistory).toContain('synthetic reasoning')
    expect(serializedHistory).not.toContain('private correction preamble')
    expect(serializedHistory).not.toContain('{"amount":123,}')
    const blocks = result.agentMessages.flatMap((message) => message.content)
    expect(blocks.filter((block) => 'toolUse' in block)).toEqual([{ toolUse: { name: 'transfer', toolUseId: 'failed', input: { amount: 123 } } }])
    expect(blocks.filter((block) => 'toolResult' in block)).toMatchObject([{ toolResult: { toolUseId: 'failed', status: 'success' } }])
  })

  it.each([false, true])('preserves normal requests and event ordering (stream=%s)', async (streaming) => {
    const baseline = await run('normal', streaming, false)
    const guarded = await run('normal', streaming, true, false)
    expect(guarded.failure).toBeUndefined()
    expect(guarded.requests).toEqual(baseline.requests)
    expect(guarded.observed).toEqual(baseline.observed)
    expect(guarded.sequence).toEqual(baseline.sequence)
    expect(guarded.executions).toEqual([123])
  })

  it.each(['mixed', 'length', 'repeat'] as const)('does not execute an invalid or ambiguous call (%s)', async (scenario) => {
    const result = await run(scenario, false)
    expect(result.failure).toBeDefined()
    expect(result.executions).toEqual([])
    expect(result.requests).toHaveLength(scenario === 'repeat' ? 2 : 1)
    expect(inspect(result.failure)).not.toContain('{"amount":123,}')
  })

  it('does not send encoded credentials back in the recovery request', async () => {
    const result = await run('secret', false)
    expect(result.failure).toBeDefined()
    expect(String(result.failure)).not.toContain('fixture')
    expect(result.requests).toHaveLength(1)
    expect(result.executions).toEqual([])
  })

  it('does not retry or run the tool when the invocation is cancelled', async () => {
    const result = await run('cancel', false)
    expect(result.requests).toHaveLength(1)
    expect(result.executions).toEqual([])
  })

  it('never publishes the tool when cancellation races with the correction response', async () => {
    const result = await run('cancel-correction', true)
    expect(result.requests).toHaveLength(2)
    expect(result.exposedBeforeRetry).toBeGreaterThan(0)
    expect(result.executions).toEqual([])
    expect(result.observed).not.toContain('modelMessageStopEvent')
  })

  it.each([false, true])('leaves correction of incomplete JSON to the model (stream=%s)', async (streaming) => {
    const baseline = await run('partial', streaming, false)
    expect(baseline.failure).toBeDefined()
    expect(baseline.requests).toHaveLength(1)
    expect(baseline.executions).toEqual([])

    const result = await run('partial', streaming)
    expect(result.failure).toBeUndefined()
    const feedback = result.requests[1]?.messages.find((message: any) => message.role === 'tool')
    expect(JSON.parse(feedback.content.split('\n')[1]).raw_arguments).toBe('{"amount":12')
    expect(result.executions).toEqual([123])
  })

  it('delimits multiline arguments and reports the original character location', async () => {
    const result = await run('multiline', true)
    expect(result.failure).toBeUndefined()
    const content = String(result.requests[1]?.messages.find((message: any) => message.role === 'tool').content)
    const lines = content.split('\n')
    expect(lines).toHaveLength(3)
    expect(JSON.parse(lines[1]!)).toEqual({
      raw_arguments: multilineArguments,
      parse_error: expect.stringMatching(/\(line 2, column 28\)$/),
    })
    expect(result.executions).toEqual([123])
  })

  it('marks a failed attempt with missing usage without inventing usage', async () => {
    const result = await run('usage_missing', false)
    expect(result.failure).toBeUndefined()
    expect(result.metadata[0].recovery.failedAttempts).toEqual([{}])
    expect(result.sdkUsage[0]).toMatchObject({ inputTokens: 9, outputTokens: 6, totalTokens: 15 })
  })

  it('uses a named correction after an explicitly required initial call', async () => {
    const result = await run('malformed', false, true, true)
    expect(result.failure).toBeUndefined()
    expect(result.requests[0]?.tool_choice).toBe('required')
    expect(result.requests[1]?.tool_choice).toEqual(namedChoice)
    expect(result.executions).toEqual([123])
  })

  it.each([false, true])('falls back to required only for explicit named-choice rejection (stream=%s)', async (stream) => {
    const recovered = await run('fallback', stream)
    expect(recovered.failure).toBeUndefined()
    expect(recovered.requests.map((request) => request.tool_choice)).toEqual([undefined, namedChoice, 'required', undefined])
    expect(recovered.requests[1]?.messages).toEqual(recovered.requests[2]?.messages)
    expect(recovered.executions).toEqual([123])
    const failed = await run('bad-request', stream)
    expect(failed.failure).toBeDefined()
    expect(failed.requests).toHaveLength(2)
    expect(failed.executions).toEqual([])
  })

  it.each(['no-tool', 'wrong-tool', 'extra-tool', 'blank', 'array'] as const)(
    'rejects an invalid correction before emitting its tool or text (%s)', async (scenario) => {
      for (const stream of [false, true]) {
        const result = await run(scenario, stream)
        expect(result.failure).toBeDefined()
        expect(result.observed).not.toContain('modelMessageStopEvent')
        expect(result.executions).toEqual([])
        expect(result.requests).toHaveLength(2)
      }
    },
  )

  it('honors the request timeout while buffered events are being consumed', async () => {
    const model = new ExaoneModel({ apiKey: 'fixture', stream: false, timeoutMs: 40,
      fetch: async () => new Response(JSON.stringify({ choices: [{ index: 0, message: { content: 'ok' }, finish_reason: 'stop' }] })) })
    const iterator = model.recoveryModel().stream([Message.fromMessageData({ role: 'user', content: [{ text: 'hi' }] })])
    expect((await iterator.next()).done).toBe(false)
    await new Promise((resolve) => setTimeout(resolve, 70))
    await expect(iterator.next()).rejects.toMatchObject({ name: 'TimeoutError' })
  })

  it('honors cancellation while buffered events are being consumed', async () => {
    const controller = new AbortController()
    const model = new ExaoneModel({ apiKey: 'fixture', stream: false,
      fetch: async () => new Response(JSON.stringify({ choices: [{ index: 0, message: { content: 'ok' }, finish_reason: 'stop' }] })) })
    const iterator = model.recoveryModel().stream([Message.fromMessageData({ role: 'user', content: [{ text: 'hi' }] })], { cancelSignal: controller.signal })
    expect((await iterator.next()).done).toBe(false)
    controller.abort()
    await expect(iterator.next()).rejects.toMatchObject({ name: 'AbortError' })
  })

  it.each([false, true])('persists and restores only the corrected call using a file-backed SDK session (stream=%s)', async (streaming) => {
    const directory = await mkdtemp(join(tmpdir(), 'exaone-tool-recovery-session-'))
    try {
      const storage = new ObservedFileStorage(directory)
      const session = new SessionManager({ storage, sessionId: 'recovery-session', saveLatestOn: 'message' })
      const first = await run('malformed', streaming, true, false, session)
      expect(first.failure).toBeUndefined()
      expect(first.executions).toEqual([123])
      expect(storage.writes.length).toBeGreaterThan(1)
      for (const snapshot of storage.writes) {
        expect(snapshot).not.toContain('123,}')
      }

      const keys = await storage.list('')
      expect(keys.length).toBeGreaterThan(0)
      const stored = await Promise.all(keys.map(async (key) => new TextDecoder().decode(await storage.read(key) ?? new Uint8Array())))
      for (const snapshot of stored) {
        expect(snapshot).not.toContain('123,}')
      }
      expect(stored.some((snapshot) => snapshot.includes('failed'))).toBe(true)

      const resumedRequests: Record<string, any>[] = []
      const resumedExecutions: number[] = []
      const record = tool({ name: 'transfer', description: 'Record a synthetic value', inputSchema: z.object({ amount: z.number().int() }),
        callback: ({ amount }) => { resumedExecutions.push(amount); return 'recorded' } })
      const fetcher: typeof fetch = async (_input, init) => {
        resumedRequests.push(JSON.parse(String(init?.body)) as Record<string, any>)
        return new Response(JSON.stringify({ choices: [{ index: 0, message: { content: 'resumed' }, finish_reason: 'stop' }] }))
      }
      const resumedStorage = new ObservedFileStorage(directory)
      const restoredSession = new SessionManager({ storage: resumedStorage, sessionId: 'recovery-session', saveLatestOn: 'message' })
      const resumed = new Agent({ id: 'recovery-agent', model: new ExaoneModel({ apiKey: 'fixture', fetch: fetcher, stream: false }),
        tools: [record], systemPrompt: 'Unchanged system instruction', printer: false, retryStrategy: null, sessionManager: restoredSession })
      resumed.addMiddleware(InvokeModelStage, toolCallRecovery())
      await resumed.invoke('Continue after resuming.')

      expect(resumedRequests).toHaveLength(1)
      const messages = resumedRequests[0]?.messages
      expect(messages.filter((message: any) => message.role === 'user').map((message: any) => message.content))
        .toEqual(['Call transfer using amount 123', 'Continue after resuming.'])
      const calls = messages.filter((message: any) => message.role === 'assistant').flatMap((message: any) => message.tool_calls ?? [])
      expect(calls).toEqual([call('failed', '{"amount":123}')])
      expect(messages.filter((message: any) => message.role === 'tool')).toMatchObject([{ tool_call_id: 'failed' }])
      expect(JSON.stringify(messages)).not.toContain('123,}')
      expect(resumedExecutions).toEqual([])
      expect(resumedStorage.writes.length).toBeGreaterThan(0)
      for (const snapshot of resumedStorage.writes) {
        expect(snapshot).not.toContain('123,}')
      }
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })
})
