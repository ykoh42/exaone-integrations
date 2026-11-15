/** Opt-in live integration checks. Set EXAONE_API_KEY in the environment. */
import assert from 'node:assert/strict'
import { Agent, Message, tool } from '@strands-agents/sdk'
import { OpenAIModel } from '@strands-agents/sdk/models/openai'
import { z } from 'zod'
import { ExaoneModel } from '../packages/strands-exaone-ts/dist/index.js'

if (!process.env.EXAONE_API_KEY) throw new Error('Set EXAONE_API_KEY to run this opt-in live test')
const requests = []
const metadata = []
let lastRequest = 0
let toolCalls = 0
const fetchWithProbe = async (url, init) => {
  const wait = Math.max(0, 1600 - (Date.now() - lastRequest))
  if (wait) await new Promise((resolve) => setTimeout(resolve, wait))
  lastRequest = Date.now()
  const body = JSON.parse(String(init.body))
  const resultsById = new Map(body.messages
    .filter((m) => m.role === 'tool' && typeof m.tool_call_id === 'string' && m.tool_call_id.length > 0)
    .map((m) => [m.tool_call_id, m.content]))
  const expectedResults = { get_test_value: '7', first_number: '3', second_number: '4' }
  const verifiedToolResults = body.messages
    .filter((m) => m.role === 'assistant')
    .flatMap((m) => m.tool_calls ?? [])
    .filter((call) => Object.hasOwn(expectedResults, call.function?.name)
      && resultsById.get(call.id) === expectedResults[call.function.name])
    .map((call) => call.function?.name)
  requests.push({
    stream: body.stream,
    preserveThinking: body.chat_template_kwargs?.preserve_thinking === true,
    reasoningHistory: body.messages.some((m) => m.role === 'assistant' && Boolean(m.reasoning_content)),
    maxToolCalls: Math.max(0, ...body.messages.map((m) => m.tool_calls?.length ?? 0)),
    verifiedToolResults,
  })
  return globalThis.fetch(url, init)
}
const report = (testCase, fields) => console.log(JSON.stringify({ case: testCase, ...fields }))
async function aggregate(model, message, options) {
  const iterator = model.streamAggregated([Message.fromMessageData({ role: 'user', content: [{ text: message }] })], options)
  while (true) {
    const next = await iterator.next()
    if (next.done) return next.value
  }
}
const model = new ExaoneModel({
  reasoningEffort: 'low', preserveThinking: true, maxTokens: 512, fetch: fetchWithProbe,
  maxRetries: 12, baseRetryDelayMs: 10000, maxRetryDelayMs: 120000,
  onMetadata: (item) => metadata.push({ reasoningTokens: Number.isInteger(item.usage?.reasoning_tokens), requestId: Boolean(item.requestId) }),
})
const streamed = await aggregate(model, 'Say a short greeting.')
assert.equal(requests.at(-1).stream, true)
assert(streamed.message.content.some((b) => b.type === 'reasoningBlock' && b.text))
assert(streamed.message.content.some((b) => b.type === 'textBlock' && b.text))
report('stream', { text: true, reasoning: true })
model.updateConfig({ stream: false })
const nonstreamed = await aggregate(model, 'Say a short greeting.')
assert.equal(requests.at(-1).stream, false)
assert(nonstreamed.message.content.some((b) => b.type === 'textBlock' && b.text))
report('nonstream', { text: true })
model.updateConfig({ stream: true })
const conversation = new Agent({ model, printer: false, retryStrategy: null })
await conversation.invoke('Remember the code word amber. Reply with one short sentence.')
await conversation.invoke('Repeat the code word I asked you to remember.')
assert(requests.at(-1).reasoningHistory && requests.at(-1).preserveThinking)
report('reasoning_history', { forwarded: true })
const lookup = tool({ name: 'get_test_value', description: 'Return the fixed test value seven.', inputSchema: z.object({}), callback: () => { toolCalls++; return '7' } })
const toolStart = requests.length
await new Agent({ model, tools: [lookup], printer: false, retryStrategy: null }).invoke('First call the get_test_value tool, then report the returned value. Do not guess.')
assert(toolCalls > 0 && requests.slice(toolStart).some((item) => item.verifiedToolResults.includes('get_test_value')))
report('agent_tool_roundtrip', { toolCalls, toolResultForwarded: true })
const parallelCalls = []
const first = tool({ name: 'first_number', description: 'Return first fixed number.', inputSchema: z.object({}), callback: () => { parallelCalls.push('first'); return '3' } })
const second = tool({ name: 'second_number', description: 'Return second fixed number.', inputSchema: z.object({}), callback: () => { parallelCalls.push('second'); return '4' } })
const parallelModel = new ExaoneModel({ reasoningEffort: 'low', preserveThinking: true, maxTokens: 512, params: { parallel_tool_calls: true }, fetch: fetchWithProbe, maxRetries: 12, baseRetryDelayMs: 10000, maxRetryDelayMs: 120000 })
const parallelStart = requests.length
await new Agent({ model: parallelModel, tools: [first, second], printer: false, retryStrategy: null }).invoke('You must call both first_number and second_number before adding their returned values. Do not guess.', { limits: { turns: 4 } })
const parallelRequests = requests.slice(parallelStart)
assert.deepEqual([...new Set(parallelCalls)].sort(), ['first', 'second'])
assert(parallelRequests.some((item) => ['first_number', 'second_number'].every((name) => item.verifiedToolResults.includes(name))))
// The model may call tools one at a time even when parallel calls are enabled.
const parallelCallsObserved = parallelRequests.some((item) => item.maxToolCalls >= 2)
report('parallel_tool_agent', { bothToolsCalled: true, resultsForwarded: 2, parallelCallsObserved })
const structured = await aggregate(model, 'Return the value count as 3.', { structuredOutputSchema: z.object({ count: z.number() }) })
const text = structured.message.content.filter((b) => b.type === 'textBlock').map((b) => b.text).join('')
assert.equal(z.object({ count: z.number() }).parse(JSON.parse(text)).count, 3)
report('structured_output', { validated: true })
const schemaAgent = new Agent({ model, structuredOutputSchema: z.object({ count: z.number() }), printer: false, retryStrategy: null })
const schemaResult = await schemaAgent.invoke('Return the value count as 3.', { limits: { turns: 3, structuredOutputAttempts: 2 } })
assert.equal(schemaResult.structuredOutput?.count, 3)
report('agent_structured_output', { validated: true })
assert(metadata.some((item) => item.reasoningTokens))
report('metadata', { requests: requests.length, reasoningTokens: true, requestIds: metadata.every((item) => item.requestId) })

// Identical two-turn configuration, but using the SDK's OpenAI chat provider.
const { retry429Fetch } = await import('./live-throttle.mjs')
const baselineStart = requests.length
const baseline = new OpenAIModel({
  api: 'chat', modelId: 'k-exaone_v2.5', apiKey: process.env.EXAONE_API_KEY, maxTokens: 512,
  clientConfig: { baseURL: 'https://api.exaone.ai/v1', maxRetries: 0, fetch: retry429Fetch(fetchWithProbe) },
  params: { reasoning_effort: 'low', chat_template_kwargs: { preserve_thinking: true } },
})
const baselineAgent = new Agent({ model: baseline, printer: false, retryStrategy: null })
await baselineAgent.invoke('Remember the code word amber. Reply with one short sentence.')
await baselineAgent.invoke('Repeat the code word I asked you to remember.')
report('openai_baseline_reasoning_history', { requests: requests.length - baselineStart, forwarded: requests.at(-1).reasoningHistory, preserveThinking: requests.at(-1).preserveThinking })
