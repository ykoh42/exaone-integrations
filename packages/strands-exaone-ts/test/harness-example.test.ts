import { execFile } from 'node:child_process'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { describe, expect, it } from 'vitest'

const run = promisify(execFile)
const example = fileURLToPath(new URL('../examples/harness/agent.ts', import.meta.url))
const chunk = (delta: unknown, finishReason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason: finishReason }] })
const sse = (chunks: unknown[]) => chunks.map((value) => `data: ${JSON.stringify(value)}\n\n`).join('') + 'data: [DONE]\n\n'

/** Runs the example with local synthetic responses, without external services. */
describe('Harness example', () => {
  it.each([
    { allowFileRead: false, webSearchEnabled: false },
    { allowFileRead: false, webSearchEnabled: true },
    { allowFileRead: true, webSearchEnabled: false },
    { allowFileRead: true, webSearchEnabled: true },
  ])('runs with file read=$allowFileRead and optional Exa search=$webSearchEnabled', async ({ allowFileRead, webSearchEnabled }) => {
    const dir = await mkdtemp(join(tmpdir(), 'exaone-example-'))
    const image = join(dir, 'synthetic.png')
    const requests: Record<string, unknown>[] = []
    const server = createServer(async (request, reply) => {
      let body = ''
      for await (const part of request) body += String(part)
      requests.push(JSON.parse(body) as Record<string, unknown>)
      const chunks = requests.length === 1 && allowFileRead ? [
        chunk({ reasoning_content: 'synthetic reasoning' }),
        chunk({ tool_calls: [{ index: 0, id: 'read-1', type: 'function', function: { name: 'read', arguments: JSON.stringify({ path: image }) } }] }),
        chunk({}, 'tool_calls'),
      ] : [chunk({ reasoning_content: 'synthetic reasoning', content: 'Synthetic answer.' }), chunk({}, 'stop')]
      reply.writeHead(200, { 'content-type': 'text/event-stream' })
      reply.end(sse(chunks))
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const address = server.address()
    if (address === null || typeof address === 'string') throw new Error('No local test port')
    try {
      await writeFile(image, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/aXcAAAAASUVORK5CYII=', 'base64'))
      const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !/KEY|TOKEN|SECRET/i.test(name) && !name.startsWith('AWS_')))
      const args = [
        '--import',
        'tsx',
        example,
        ...(allowFileRead ? ['--allow-file-read'] : []),
        'Read the synthetic image',
      ]
      const { stdout } = await run(process.execPath, args, { env: {
        ...env,
        EXAONE_API_KEY: 'test-key',
        EXAONE_BASE_URL: `http://127.0.0.1:${address.port}/v1`,
        EXA_WEB_SEARCH: webSearchEnabled ? 'true' : 'false',
        NO_PROXY: '127.0.0.1,localhost,::1',
      } })
      expect(stdout).toContain('Synthetic answer.')
      expect(stdout).not.toContain('synthetic reasoning')
      expect(requests).toHaveLength(allowFileRead ? 2 : 1)
      const tools = requests[0]!.tools as { function: { name: string } }[]
      expect(tools.some((item) => item.function.name === 'read')).toBe(allowFileRead)
      expect(tools.some((item) => item.function.name === 'web_search')).toBe(webSearchEnabled)
      if (allowFileRead) {
        const messages = requests[1]!.messages as Record<string, unknown>[]
        expect(messages).toContainEqual(expect.objectContaining({ role: 'tool', tool_call_id: 'read-1', content: expect.stringContaining('contents are not shown') }))
      }
    } finally {
      await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
      await rm(dir, { recursive: true, force: true })
    }
  })
})
