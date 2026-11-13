import { Agent } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/model.js'

describe('EXAONE model info', () => {
  it('provides known capabilities without inventing output or caching limits', () => {
    const model = new ExaoneModel()
    expect(model.modelInfo).toEqual({
      limits: { contextWindow: 524_288 },
      capabilities: { stateful: false, forcedToolChoice: true, nativeStructuredOutput: true },
    })
    expect(model.getConfig().modelInfo).toBeUndefined()
  })

  it('preserves false overrides and the SDK limit precedence', () => {
    const model = new ExaoneModel({
      contextWindowLimit: 4096,
      modelInfo: { capabilities: { nativeStructuredOutput: false } },
    })
    expect(model.modelInfo.limits?.contextWindow).toBe(4096)
    expect(model.modelInfo.capabilities?.nativeStructuredOutput).toBe(false)
    expect(model.modelInfo.capabilities?.forcedToolChoice).toBe(true)
    model.updateConfig({ modelInfo: {
      limits: { contextWindow: 8192, maxOutputTokens: 1024 },
      capabilities: { stateful: true, forcedToolChoice: false },
    } })
    expect(model.modelInfo.limits).toEqual({ contextWindow: 8192, maxOutputTokens: 1024 })
    expect(model.modelInfo.capabilities).toMatchObject({
      stateful: true, forcedToolChoice: false, nativeStructuredOutput: true,
    })
  })

  it('treats undefined overrides as inheritance', () => {
    const model = new ExaoneModel({ contextWindowLimit: undefined, modelInfo: {
      limits: { contextWindow: undefined },
      capabilities: { forcedToolChoice: undefined, nativeStructuredOutput: undefined },
    } })
    expect(model.modelInfo.capabilities?.forcedToolChoice).toBe(true)
    expect(model.modelInfo.limits?.contextWindow).toBe(524_288)
    expect(model.modelInfo.capabilities?.nativeStructuredOutput).toBe(true)
  })

  it('recomputes facts when the model changes', () => {
    const model = new ExaoneModel()
    model.updateConfig({ modelId: 'unknown-model' })
    expect(model.modelInfo.capabilities?.nativeStructuredOutput).toBeUndefined()
    expect(model.modelInfo.capabilities?.forcedToolChoice).toBeUndefined()
    model.updateConfig({ modelId: 'k-exaone_v2.5' })
    expect(model.modelInfo.capabilities?.nativeStructuredOutput).toBe(true)
  })

  it('preserves the SDK agent structured-output contract', async () => {
    const bodies: Record<string, unknown>[] = []
    const fetcher = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)))
      return Response.json({ choices: [{
        message: { role: 'assistant', content: '{"name":"Alex"}' }, finish_reason: 'stop',
      }] })
    }) as typeof fetch
    const agent = new Agent({
      model: new ExaoneModel({ apiKey: 'synthetic', fetch: fetcher, stream: false }),
      structuredOutputSchema: z.object({ name: z.string() }),
      printer: false,
    })
    try {
      const result = await agent.invoke('Extract Alex.')
      expect(result.structuredOutput).toEqual({ name: 'Alex' })
      expect(bodies[0]?.tools).toEqual(expect.arrayContaining([expect.objectContaining({
        function: expect.objectContaining({ name: 'strands_structured_output' }),
      })]))
    } finally {
      await agent.shutdown()
    }
  })
})
