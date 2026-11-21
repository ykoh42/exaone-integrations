/** Opt-in, serial provider checks against an already-running Chat Completions server. */
import { Agent, ImageBlock, Message, TextBlock, tool as strandsTool } from '@strands-agents/sdk'
import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool as langchainTool } from '@langchain/core/tools'
import { z } from 'zod'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { ExaoneModel } from 'strands-exaone'
import { ChatExaone } from 'langchain-exaone'

const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAIAAABvFaqvAAAAIElEQVR4nGP4z8BAFUQdU0YNGjVo1KBRg0YNGjWIIgQABMo93z5bbNQAAAAASUVORK5CYII=', 'base64')
const names = new Set(['framework', 'base-url', 'model', 'thinking', 'server', 'server-version', 'model-revision', 'reasoning-parser', 'tool-parser', 'server-options', 'max-tokens'])
const args = {}
for (let i = 2; i < process.argv.length; i++) {
  const name = process.argv[i].replace(/^--/, '')
  if (name === 'image') { args.image = true; continue }
  if (!process.argv[i].startsWith('--') || !names.has(name) || !process.argv[i + 1] || process.argv[i + 1].startsWith('--')) throw Error('Invalid option')
  args[name] = process.argv[++i]
}
for (const name of ['framework', 'base-url', 'model', 'thinking']) if (!args[name]) throw Error('Missing required option')
if (!['strands', 'langchain'].includes(args.framework) || !['on', 'off'].includes(args.thinking) || (args.server && !['vllm', 'sglang', 'unknown'].includes(args.server))) throw Error('Invalid option')
const maxTokens = Number(args['max-tokens'] ?? 1024)
if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0) throw Error('Invalid max tokens')
const apiKey = process.env.SELF_HOSTED_API_KEY ?? 'EMPTY'
const require = createRequire(import.meta.url)
function packageVersion(name) {
  let path = dirname(require.resolve(name))
  while (true) {
    try {
      const data = JSON.parse(readFileSync(join(path, 'package.json'), 'utf8'))
      if (data.name === name) return data.version
    } catch {}
    const parent = dirname(path)
    if (parent === path) throw Error('Package version unavailable')
    path = parent
  }
}
const requests = []
let currentCase = 'setup'
const fetchWithAudit = async (url, init) => {
  const body = JSON.parse(String(init.body))
  const callIds = new Set(body.messages.filter((m) => m.role === 'assistant')
    .flatMap((m) => m.tool_calls ?? []).filter((c) => c.function?.name === 'lookup').map((c) => c.id))
  requests.push({
    stream: body.stream,
    reasoningHistory: body.messages.some((m) => m.role === 'assistant' && Boolean(m.reasoning_content || m.reasoning)),
    toolResult: body.messages.some((m) => m.role === 'tool' && m.content === '7' && callIds.has(m.tool_call_id)),
  })
  return globalThis.fetch(url, init)
}
const check = (condition) => { if (!condition) throw Error('Smoke assertion failed') }
const report = (name) => console.log(JSON.stringify({ case: name, status: 'passed' }))
const userText = 'Reply with the single word READY.'
const memoryText = 'Remember the word amber. Reply briefly.'
const followupText = 'Repeat the word I asked you to remember.'
const toolText = 'Call lookup with value 7, then report its result.'
const imageText = 'What is the main color of this image? Answer in English.'

async function strands() {
  const model = new ExaoneModel({
    baseURL: args['base-url'], modelId: args.model, apiKey, enableThinking: args.thinking === 'on',
    maxTokens, includeUsage: false, fetch: fetchWithAudit,
  })
  for (const stream of [false, true]) {
    currentCase = stream ? 'text_stream' : 'text_nonstream'
    model.updateConfig({ stream })
    const iterator = model.streamAggregated([Message.fromMessageData({ role: 'user', content: [{ text: userText }] })])
    let result
    while (true) {
      const next = await iterator.next()
      if (next.done) { result = next.value; break }
    }
    check(result.message.content.some((b) => b.type === 'textBlock' && b.text?.trim()))
    check(result.stopReason === 'endTurn')
    check(requests.at(-1).stream === stream)
    check(result.message.content.some((b) => b.type === 'reasoningBlock' && b.text) === (args.thinking === 'on'))
    report(stream ? 'text_stream' : 'text_nonstream')
  }
  model.updateConfig({ stream: true })
  currentCase = 'multiturn'
  const conversation = new Agent({ model, printer: false, retryStrategy: null })
  const first = await conversation.invoke(memoryText)
  check(first.stopReason === 'endTurn')
  const answer = await conversation.invoke(followupText)
  check(answer.stopReason === 'endTurn')
  check(answer.lastMessage.content.some((b) => b.type === 'textBlock' && b.text.toLowerCase().includes('amber')))
  if (args.thinking === 'on') check(requests.at(-1).reasoningHistory)
  report('multiturn')
  const used = []
  const lookup = strandsTool({
    name: 'lookup', description: 'Return a fixed test value.',
    inputSchema: z.object({ value: z.number().int() }),
    callback: ({ value }) => { used.push(value); return '7' },
  })
  for (const stream of [false, true]) {
    currentCase = stream ? 'tool_stream' : 'tool_nonstream'
    model.updateConfig({ stream })
    used.length = 0
    const start = requests.length
    const result = await new Agent({ model, tools: [lookup], printer: false, retryStrategy: null }).invoke(toolText, { limits: { turns: 4 } })
    check(result.stopReason === 'endTurn')
    check(used.length === 1 && used[0] === 7 && requests.slice(start).some((r) => r.toolResult))
    report(stream ? 'tool_stream' : 'tool_nonstream')
  }
  if (args.image) {
    currentCase = 'image'
    const result = await new Agent({ model, printer: false, retryStrategy: null }).invoke([
      new TextBlock(imageText), new ImageBlock({ format: 'png', source: { bytes: image } }),
    ])
    check(result.stopReason === 'endTurn')
    check(result.lastMessage.content.some((b) => b.type === 'textBlock' && /\bred\b/i.test(b.text)))
    report('image')
  }
}

async function langchain() {
  const model = new ChatExaone({
    baseUrl: args['base-url'], model: args.model, apiKey, enableThinking: args.thinking === 'on',
    maxTokens, fetch: fetchWithAudit,
  })
  const user = new HumanMessage(userText)
  for (const stream of [false, true]) {
    currentCase = stream ? 'text_stream' : 'text_nonstream'
    let result
    if (stream) {
      for await (const chunk of await model.stream([user])) result = result ? result.concat(chunk) : chunk
    } else result = await model.invoke([user])
    check(typeof result?.content === 'string' && result.content.trim() && requests.at(-1).stream === stream)
    check(result.response_metadata.finish_reason === 'stop')
    check(Boolean(result.additional_kwargs.reasoning_content) === (args.thinking === 'on'))
    report(stream ? 'text_stream' : 'text_nonstream')
  }
  currentCase = 'multiturn'
  const firstUser = new HumanMessage(memoryText)
  const first = await model.invoke([firstUser])
  check(first.response_metadata.finish_reason === 'stop')
  const answer = await model.invoke([firstUser, first, new HumanMessage(followupText)])
  check(answer.response_metadata.finish_reason === 'stop')
  check(typeof answer.content === 'string' && answer.content.toLowerCase().includes('amber'))
  if (args.thinking === 'on') check(requests.at(-1).reasoningHistory)
  report('multiturn')
  const lookup = langchainTool(({ value }) => String(value), {
    name: 'lookup', description: 'Return a fixed test value.', schema: z.object({ value: z.number().int() }),
  })
  const toolUser = new HumanMessage(toolText)
  const bound = model.bindTools([lookup])
  for (const stream of [false, true]) {
    currentCase = stream ? 'tool_stream' : 'tool_nonstream'
    let call
    if (stream) {
      for await (const chunk of await bound.stream([toolUser])) call = call ? call.concat(chunk) : chunk
    } else call = await bound.invoke([toolUser])
    check(call.response_metadata.finish_reason === 'tool_calls')
    if (stream) {
      check(call.tool_call_chunks?.length === 1)
      const rawArgs = JSON.parse(call.tool_call_chunks[0].args)
      check(rawArgs && typeof rawArgs === 'object' && !Array.isArray(rawArgs) && Object.keys(rawArgs).length === 1 && rawArgs.value === 7)
    }
    check(call.tool_calls?.length === 1 && call.tool_calls[0].name === 'lookup' && call.tool_calls[0].args.value === 7)
    const result = await lookup.invoke(call.tool_calls[0].args)
    const answer = await model.invoke([toolUser, call, new ToolMessage({ content: result, tool_call_id: call.tool_calls[0].id })])
    check(answer.response_metadata.finish_reason === 'stop')
    check(requests.at(-1).toolResult)
    report(stream ? 'tool_stream' : 'tool_nonstream')
  }
  if (args.image) {
    currentCase = 'image'
    const result = await model.invoke([new HumanMessage({ content: [
      { type: 'text', text: imageText },
      { type: 'image_url', image_url: { url: `data:image/png;base64,${image.toString('base64')}` } },
    ] })])
    check(result.response_metadata.finish_reason === 'stop')
    check(typeof result.content === 'string' && /\bred\b/i.test(result.content))
    report('image')
  }
}

console.log(JSON.stringify({
  framework: args.framework, language: 'typescript', server: args.server ?? 'unknown', server_version: args['server-version'] ?? 'unknown',
  model: args.model, model_revision: args['model-revision'] ?? 'unknown', reasoning_parser: args['reasoning-parser'] ?? 'unknown',
  tool_parser: args['tool-parser'] ?? 'unknown', thinking: args.thinking, image_requested: Boolean(args.image),
  server_options: args['server-options'] ?? '',
  provider_version: packageVersion(`${args.framework}-exaone`),
  sdk_version: packageVersion(args.framework === 'strands' ? '@strands-agents/sdk' : '@langchain/core'),
}))
try {
  await (args.framework === 'strands' ? strands() : langchain())
} catch (error) {
  console.error(JSON.stringify({ case: currentCase, status: 'failed', error_type: error?.constructor?.name ?? 'Error' }))
  process.exitCode = 1
}
