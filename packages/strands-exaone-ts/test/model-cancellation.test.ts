import { createHarness } from '@strands-agents/harness'
import { Message, tool } from '@strands-agents/sdk'
import type { ModelStreamEvent } from '@strands-agents/sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ExaoneModel } from '../src/index.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const toolCall = { index: 0, id: 'call_1', type: 'function', function: { name: 'lookup', arguments: '{}' } }
const makeResponse = (streaming: boolean) => {
  const message = { content: 'visible', tool_calls: [toolCall] }
  if (!streaming) return new Response(JSON.stringify({ choices: [{ index: 0, message, finish_reason: 'tool_calls' }] }))
  return new Response(`data: ${JSON.stringify({ choices: [{ index: 0, delta: message, finish_reason: 'tool_calls' }] })}\n\ndata: [DONE]\n\n`)
}
const options = { builtinTools: [] as const, builtinPlugins: [] as const, session: false as const, memory: false as const, skills: false as const, caching: false as const, printer: false as const, retryStrategy: null }
afterEach(() => vi.useRealTimers())

describe('cancellation between model events', () => {
  it.each([true, false])('does not emit the first event if cancellation occurs while fetching (stream=%s)', async (streaming) => {
    const controller = new AbortController()
    const callback = vi.fn()
    const model = new ExaoneModel({ apiKey: 'key', stream: streaming, onMetadata: callback, fetch: async () => {
      controller.abort(new Error('cancelled'))
      return makeResponse(streaming)
    } })
    const events = []
    await expect((async () => { for await (const event of model.stream([user()], { cancelSignal: controller.signal })) events.push(event) })()).rejects.toThrow('cancelled')
    expect(events).toEqual([])
    expect(callback).not.toHaveBeenCalled()
  })

  it.each([true, false])('does not emit buffered events after cancellation (stream=%s)', async (streaming) => {
    for (const boundary of ['messageStart', 'textStart', 'textDelta', 'toolStart', 'messageStop']) {
      const controller = new AbortController()
      const callback = vi.fn()
      const model = new ExaoneModel({ apiKey: 'key', stream: streaming, fetch: async () => makeResponse(streaming), onMetadata: callback })
      const events: string[] = []
      const isBoundary = (event: ModelStreamEvent): boolean => boundary === 'messageStart' ? event.type === 'modelMessageStartEvent'
        : boundary === 'textStart' ? event.type === 'modelContentBlockStartEvent' && !event.start
          : boundary === 'textDelta' ? event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta'
            : boundary === 'toolStart' ? event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart'
              : event.type === 'modelMessageStopEvent'
      await expect((async () => {
        for await (const event of model.stream([user()], { cancelSignal: controller.signal })) {
          events.push(event.type)
          if (isBoundary(event)) controller.abort(new Error('cancelled'))
          else expect(controller.signal.aborted).toBe(false)
        }
      })()).rejects.toThrow('cancelled')
      expect(controller.signal.aborted).toBe(true)
      expect(events).not.toContain('modelMetadataEvent')
      expect(callback).not.toHaveBeenCalled()
    }
  })

  it('stops buffered tool events on timeout', async () => {
    vi.useFakeTimers()
    const model = new ExaoneModel({ apiKey: 'key', timeoutMs: 100, fetch: async () => makeResponse(true) })
    const events: string[] = []
    await expect((async () => {
      for await (const event of model.stream([user()])) {
        events.push(event.type)
        if (event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart') await vi.advanceTimersByTimeAsync(100)
      }
    })()).rejects.toMatchObject({ name: 'TimeoutError' })
    expect(events.at(-1)).toBe('modelContentBlockStartEvent')
    expect(events).not.toContain('modelMessageStopEvent')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not execute a tool when Harness cancels at its start event', async () => {
    const callback = vi.fn(() => 'unexpected')
    const lookup = tool({ name: 'lookup', description: 'Synthetic lookup', inputSchema: z.object({}), callback })
    const model = new ExaoneModel({ apiKey: 'key', fetch: async () => makeResponse(true) })
    const agent = await createHarness({ ...options, model, tools: [lookup] })
    let toolStartSeen = false
    for await (const event of agent.stream('Use the tool')) {
      if (event.type === 'modelStreamUpdateEvent' && event.event.type === 'modelContentBlockStartEvent' && event.event.start?.type === 'toolUseStart') {
        toolStartSeen = true
        agent.cancel()
      }
    }
    expect(toolStartSeen).toBe(true)
    expect(callback).not.toHaveBeenCalled()
    await agent.shutdown()
  })
})
