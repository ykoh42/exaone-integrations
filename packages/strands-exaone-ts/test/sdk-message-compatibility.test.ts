import * as sdk from '@strands-agents/sdk'
import { describe, expect, it } from 'vitest'
import { ExaoneModel } from '../src/index.js'

function captureRequests(messages: sdk.Message[] = []) {
  const requests: Record<string, unknown>[][] = []
  const fetcher: typeof fetch = async (_url, init) => {
    requests.push(JSON.parse(String(init?.body)).messages)
    return Response.json({ choices: [{ index: 0, message: { content: 'Done.' }, finish_reason: 'stop' }] })
  }
  const model = new ExaoneModel({ apiKey: 'fixture', fetch: fetcher, stream: false })
  return { agent: new sdk.Agent({ model, messages, printer: false, retryStrategy: null }), requests }
}

describe('Strands SDK session content', () => {
  it.skipIf(!('UnknownBlock' in sdk))('keeps unknown stored blocks in history and omits them from model requests', async () => {
    const futureBlock = { futureBlock: { text: 'Synthetic saved data.' } }
    const restore = (role: sdk.Role, content: (sdk.ContentBlockData | typeof futureBlock)[]) =>
      sdk.Message.fromJSON({ role, content: content as sdk.ContentBlockData[] })
    const { agent, requests } = captureRequests([
      restore('user', [{ text: 'Begin.' }, futureBlock]),
      restore('assistant', [futureBlock]),
      restore('assistant', [{ toolUse: { toolUseId: 'lookup-1', name: 'lookup', input: {} } }, futureBlock]),
      restore('user', [{
        toolResult: { toolUseId: 'lookup-1', status: 'success', content: [{ text: 'Answer.' }, futureBlock] },
      }]),
    ])
    await agent.invoke('Continue.')
    expect(requests).toEqual([[
      { role: 'user', content: 'Begin.' },
      {
        role: 'assistant', content: null,
        tool_calls: [{ id: 'lookup-1', type: 'function', function: { name: 'lookup', arguments: '{}' } }],
      },
      { role: 'tool', content: 'Answer.', tool_call_id: 'lookup-1' },
      { role: 'user', content: 'Continue.' },
    ]])
    expect(agent.messages[1]?.toJSON().content).toEqual([futureBlock])
  })

  it.skipIf(!('InjectedBlock' in sdk))('receives the SDK-rendered injection only on its current user turn', async () => {
    const first = {
      role: 'user',
      content: [{ text: 'First.' }, { injected: { text: 'SDK-provided context.', source: 'fixture' } }],
    } as sdk.MessageData
    const { agent, requests } = captureRequests()
    await agent.invoke([sdk.Message.fromJSON(first)])
    await agent.invoke('Second.')
    expect(requests[0]).toEqual([{ role: 'user', content: 'First.\n\nSDK-provided context.' }])
    expect(requests[1]).toEqual([
      { role: 'user', content: 'First.' },
      { role: 'assistant', content: 'Done.' },
      { role: 'user', content: 'Second.' },
    ])
    expect(agent.messages[0]?.toJSON().content).toEqual(first.content)
  })
})
