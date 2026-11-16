import { createServer } from 'node:http'
import { Message } from '@strands-agents/sdk'
import { describe, expect, it } from 'vitest'
import { ExaoneModel } from '../src/index.js'

describe('local HTTP transport', () => {
  it.each(['complete', 'disconnect', 'cancel'] as const)('handles %s during streaming', async (ending) => {
    const requests: unknown[] = []
    let onClosed!: () => void
    const closed = new Promise<void>((resolve) => { onClosed = resolve })
    let disconnect!: () => void
    const disconnectRequested = new Promise<void>((resolve) => { disconnect = resolve })
    const server = createServer(async (request, reply) => {
      const data: Buffer[] = []
      for await (const chunk of request) data.push(Buffer.from(chunk))
      requests.push(JSON.parse(Buffer.concat(data).toString()))
      reply.on('close', onClosed)
      reply.writeHead(200, { 'content-type': 'text/event-stream', connection: 'close', ...(ending === 'disconnect' ? { 'content-length': '99999' } : {}) })
      const frame = `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'hello' } }] })}\n\n`
      reply.write(frame.slice(0, 10))
      reply.write(frame.slice(10))
      if (ending === 'disconnect') {
        await disconnectRequested
        reply.destroy()
      }
      if (ending === 'complete') reply.end(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('No loopback address')
    const model = new ExaoneModel({ apiKey: 'fixture', baseURL: `http://127.0.0.1:${address.port}/v1` })
    const user = Message.fromMessageData({ role: 'user', content: [{ text: 'hello' }] })
    const observed: string[] = []
    const consume = async () => {
      for await (const event of model.stream([user])) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
          observed.push(event.delta.text)
          if (ending === 'disconnect') disconnect()
          if (ending === 'cancel') break
        }
      }
    }
    try {
      if (ending === 'disconnect') await expect(consume()).rejects.toThrow()
      else await consume()
      if (ending === 'cancel') await closed
      expect(observed).toEqual(['hello'])
      expect(requests).toHaveLength(1)
      expect(requests[0]).toMatchObject({ stream: true })
    } finally {
      server.closeAllConnections()
      await new Promise<void>((resolve) => server.close(() => resolve()))
    }
  }, 10_000)
})
