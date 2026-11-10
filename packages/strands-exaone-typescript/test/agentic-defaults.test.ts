import { describe, expect, it } from 'vitest'
import { Message } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const payload = JSON.stringify({ choices: [{ index: 0, message: { content: '{"count":3}' }, finish_reason: 'stop' }] })

function captureRequests(): { requests: Record<string, unknown>[]; fetch: typeof fetch } {
  const requests: Record<string, unknown>[] = []
  const fetcher: typeof fetch = async (_url, init) => {
    requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    return new Response(payload)
  }
  return { requests, fetch: fetcher }
}

async function complete(model: ExaoneModel): Promise<void> {
  for await (const _event of model.stream([user()])) { /* consume */ }
}

describe('agentic generation defaults', () => {
  it('sends the EXAONE agent recommendations when values are omitted', async () => {
    const capture = captureRequests()
    await complete(new ExaoneModel({ apiKey: 'key', stream: false, fetch: capture.fetch }))
    expect(capture.requests[0]).toMatchObject({
      temperature: 1,
      top_p: 0.95,
      chat_template_kwargs: { preserve_thinking: true },
    })
  })

  it('sends the same recommendations on the streaming path', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      const event = JSON.stringify({ choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }] })
      return new Response(`data: ${event}\n\ndata: [DONE]\n\n`)
    }
    await complete(new ExaoneModel({ apiKey: 'key', fetch: fetcher }))
    expect(requests[0]).toMatchObject({ stream: true, temperature: 1, top_p: 0.95, chat_template_kwargs: { preserve_thinking: true } })
  })

  it('preserves zero and false values in params alongside other template options', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({
      apiKey: 'key', stream: false, fetch: capture.fetch,
      params: { temperature: 0, top_p: 0, chat_template_kwargs: { preserve_thinking: false, enable_thinking: false } },
    })
    await complete(model)
    expect(capture.requests[0]).toMatchObject({
      temperature: 0,
      top_p: 0,
      chat_template_kwargs: { preserve_thinking: false, enable_thinking: false },
    })
  })

  it('treats undefined params as omitted while preserving explicit null values', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({
      apiKey: 'key', stream: false, fetch: capture.fetch,
      params: { temperature: undefined, top_p: undefined, chat_template_kwargs: { preserve_thinking: undefined } },
    })
    await complete(model)
    model.updateConfig({ params: { temperature: null, top_p: null, chat_template_kwargs: { preserve_thinking: null } } })
    await complete(model)
    expect(capture.requests[0]).toMatchObject({ temperature: 1, top_p: 0.95, chat_template_kwargs: { preserve_thinking: true } })
    expect(capture.requests[1]).toMatchObject({ temperature: null, top_p: null, chat_template_kwargs: { preserve_thinking: null } })
  })

  it('prioritizes named options over params, including zero and false', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({
      apiKey: 'key', stream: false, fetch: capture.fetch, temperature: 0, topP: 0, preserveThinking: false,
      params: { temperature: 0.8, top_p: 0.6, chat_template_kwargs: { preserve_thinking: true, enable_thinking: true } },
    })
    await complete(model)
    expect(capture.requests[0]).toMatchObject({
      temperature: 0,
      top_p: 0,
      chat_template_kwargs: { preserve_thinking: false, enable_thinking: true },
    })
  })

  it('allows named options to override false and zero values in params', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({
      apiKey: 'key', stream: false, fetch: capture.fetch, temperature: 0.5, topP: 0.8, preserveThinking: true,
      params: { temperature: 0, top_p: 0, chat_template_kwargs: { preserve_thinking: false } },
    })
    await complete(model)
    expect(capture.requests[0]).toMatchObject({ temperature: 0.5, top_p: 0.8, chat_template_kwargs: { preserve_thinking: true } })
  })

  it('uses the same precedence after updating named options and params', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({ apiKey: 'key', stream: false, fetch: capture.fetch })
    model.updateConfig({ temperature: 0, topP: 0.7, preserveThinking: false })
    await complete(model)
    model.updateConfig({ temperature: undefined, topP: undefined, preserveThinking: undefined, params: {
      temperature: 0.4, top_p: 0.6, chat_template_kwargs: { preserve_thinking: false },
    } })
    await complete(model)
    expect(capture.requests[0]).toMatchObject({ temperature: 0, top_p: 0.7, chat_template_kwargs: { preserve_thinking: false } })
    expect(capture.requests[1]).toMatchObject({ temperature: 0.4, top_p: 0.6, chat_template_kwargs: { preserve_thinking: false } })
  })

  it('uses the same defaults with structured output requests', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({ apiKey: 'key', stream: false, fetch: capture.fetch })
    for await (const _event of model.stream([user()], { structuredOutputSchema: z.object({ count: z.number() }) })) { /* consume */ }
    expect(capture.requests[0]).toMatchObject({
      temperature: 1,
      top_p: 0.95,
      chat_template_kwargs: { preserve_thinking: true },
      response_format: { type: 'json_schema' },
    })
  })

  it('preserves named overrides with structured output requests', async () => {
    const capture = captureRequests()
    const model = new ExaoneModel({
      apiKey: 'key', stream: false, fetch: capture.fetch, temperature: 0, preserveThinking: false,
      params: { top_p: 0, chat_template_kwargs: { enable_thinking: false } },
    })
    for await (const _event of model.stream([user()], { structuredOutputSchema: z.object({ count: z.number() }) })) { /* consume */ }
    expect(capture.requests[0]).toMatchObject({
      temperature: 0,
      top_p: 0,
      chat_template_kwargs: { preserve_thinking: false, enable_thinking: false },
      response_format: { type: 'json_schema' },
    })
  })
})
