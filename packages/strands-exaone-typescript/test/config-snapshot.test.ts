import { createHarness } from '@strands-agents/harness'
import { BeforeModelCallEvent, Message, ModelError, ModelNetworkError } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { ExaoneModel } from '../src/index.js'

const user = () => Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
const ok = () => new Response(JSON.stringify({ choices: [{ index: 0, message: { content: 'ok' }, finish_reason: 'stop' }] }))
async function complete(model: ExaoneModel) { for await (const _event of model.stream([user()])) { /* consume */ } }

/** Synthetic requests exercise configuration ownership without contacting EXAONE. */
describe('configuration snapshots', () => {
  it('preserves Date, null-prototype data and custom toJSON semantics', async () => {
    class CustomValue {
      #value = 'private value'
      toJSON() { return { value: this.#value } }
    }
    class CustomDate extends Date {
      #label = 'custom date'
      override toJSON() { return `${this.#label}: ${super.toJSON()}` }
    }
    class CustomArray extends Array<number> {
      #label = 'custom array'
      toJSON() { return { label: this.#label, values: [...this] } }
    }
    const custom = new CustomValue()
    const customDate = new CustomDate('2026-11-11T00:00:00.000Z')
    const customArray = new CustomArray(1, 2)
    const date = new Date('2026-11-11T00:00:00.000Z')
    const emptyPrototype = Object.assign(Object.create(null) as Record<string, unknown>, { nested: { values: [1, 2] } })
    const plain = { value: 'original', toJSON() { return { value: this.value } } }
    const params = { custom, customDate, customArray, date, emptyPrototype, plain, optional: undefined, sparse: [undefined, () => 1, null] }
    const expected = JSON.parse(JSON.stringify(params)) as Record<string, unknown>
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => { requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>); return ok() }
    const model = new ExaoneModel({ apiKey: 'test-key', stream: false, params, fetch: fetcher })
    date.setUTCFullYear(2040)
    plain.value = 'changed'
    ;(emptyPrototype.nested as { values: number[] }).values.push(3)
    const returned = model.getConfig().params!
    expect(Object.getPrototypeOf(returned.emptyPrototype)).toBeNull()
    ;(returned.date as Date).setUTCFullYear(2050)
    expect(returned.custom).toBe(custom)
    expect(returned.customDate).toBe(customDate)
    expect(returned.customArray).toBe(customArray)
    await complete(model)
    expect(requests[0]).toMatchObject(expected)
    expect(requests[0]).not.toHaveProperty('optional')
  })

  it('rejects circular values at request serialization without network retries or sensitive details', async () => {
    const cycle: { self?: unknown } = {}
    cycle.self = cycle
    const fetcher = vi.fn(async () => ok())
    const model = new ExaoneModel({ apiKey: 'test-key', stream: false, params: { cycle }, fetch: fetcher })
    const returned = model.getConfig().params!.cycle as { self: unknown }
    expect(returned.self).toBe(returned)
    const agent = await createHarness({ model, builtinTools: [], builtinPlugins: [], session: false, memory: false, skills: false, caching: false, printer: false })
    const beforeModel = vi.fn()
    agent.addHook(BeforeModelCallEvent, beforeModel)
    let failure: unknown
    try { await agent.invoke('hello') } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelError)
    expect(failure).not.toBeInstanceOf(ModelNetworkError)
    expect((failure as Error).message).toBe('EXAONE request could not be serialized as JSON')
    expect((failure as Error).cause).toBeUndefined()
    expect(beforeModel).toHaveBeenCalledOnce()
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('does not expose custom serialization exceptions or their sensitive causes', async () => {
    const bad = { toJSON() { throw new Error('test-secret and synthetic private data', { cause: new Error('more private data') }) } }
    const fetcher = vi.fn(async () => ok())
    const model = new ExaoneModel({ apiKey: 'test-secret', stream: false, params: { bad }, fetch: fetcher })
    let failure: unknown
    try { await complete(model) } catch (error) { failure = error }
    expect(failure).toBeInstanceOf(ModelError)
    expect((failure as Error).message).toBe('EXAONE request could not be serialized as JSON')
    expect((failure as Error).stack).not.toContain('test-secret')
    expect((failure as Error).stack).not.toContain('synthetic private data')
    expect((failure as Error).cause).toBeUndefined()
    expect(fetcher).not.toHaveBeenCalled()
  })

  it('isolates constructor, getConfig and updateConfig nested data while preserving callbacks', async () => {
    const requests: Record<string, unknown>[] = []
    const fetcher: typeof fetch = async (_url, init) => { requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>); return ok() }
    const onMetadata = vi.fn()
    const original = { stop: ['first'], chat_template_kwargs: { preserve_thinking: false, custom: { choices: ['initial'] } } }
    const modelInfo = { limits: { contextWindow: 5000 } }
    const model = new ExaoneModel({ apiKey: 'test-key', stream: false, params: original, modelInfo, fetch: fetcher, onMetadata })
    original.stop.push('mutated input')
    original.chat_template_kwargs.custom.choices.push('mutated input')
    modelInfo.limits.contextWindow = 1
    const exposed = model.getConfig()
    ;(exposed.params!.stop as string[]).push('mutated return')
    ;((exposed.params!.chat_template_kwargs as typeof original.chat_template_kwargs).custom.choices).push('mutated return')
    ;(exposed.modelInfo!.limits as { contextWindow: number }).contextWindow = 2
    expect(model.getConfig().fetch).toBe(fetcher)
    expect(model.getConfig().onMetadata).toBe(onMetadata)
    expect(model.getConfig().modelInfo?.limits?.contextWindow).toBe(5000)
    expect(model.getConfig()).not.toHaveProperty('apiKey')
    await complete(model)
    expect(requests[0]).toMatchObject({ stop: ['first'], chat_template_kwargs: { preserve_thinking: false, custom: { choices: ['initial'] } } })

    const update = { stop: ['updated'], chat_template_kwargs: { custom: { choices: ['new'] } } }
    model.updateConfig({ params: update })
    update.stop.push('mutated update')
    update.chat_template_kwargs.custom.choices.push('mutated update')
    await complete(model)
    expect(requests[1]).toMatchObject({ stop: ['updated'], chat_template_kwargs: { preserve_thinking: true, custom: { choices: ['new'] } } })
    expect(onMetadata).toHaveBeenCalledTimes(2)
  })

  it('serializes the same request for every retry even when external inputs change', async () => {
    const bodies: string[] = []
    const params = { stop: ['initial'], response_format: { type: 'json_object' } }
    const schema = { type: 'object', properties: { value: { type: 'string' } } }
    let model: ExaoneModel
    const fetcher: typeof fetch = async (_url, init) => {
      bodies.push(String(init?.body))
      if (bodies.length === 1) {
        params.stop.push('mutated')
        schema.properties.value.type = 'number'
        ;(model.getConfig().params!.stop as string[]).push('mutated return')
        model.updateConfig({ params: { stop: ['next request'] } })
        return new Response('{}', { status: 429, headers: { 'retry-after-ms': '0' } })
      }
      return ok()
    }
    model = new ExaoneModel({ apiKey: 'test-key', stream: false, params, fetch: fetcher, maxRetries: 1 })
    const toolSpecs = [{ name: 'lookup', description: 'Synthetic lookup', inputSchema: schema }]
    for await (const _event of model.stream([user()], { toolSpecs })) { /* consume */ }
    expect(bodies).toHaveLength(2)
    expect(bodies[1]).toBe(bodies[0])
    expect(JSON.parse(bodies[0]!)).toMatchObject({ stop: ['initial'], tools: [{ function: { parameters: { properties: { value: { type: 'string' } } } } }] })
    await complete(model)
    expect(JSON.parse(bodies[2]!)).toMatchObject({ stop: ['next request'] })
  })
})
