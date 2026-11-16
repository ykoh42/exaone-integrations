import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { Agent, tool } from '@strands-agents/sdk';
import { z } from 'zod';
import { ExaoneModel } from 'strands-exaone';

const directory = process.argv[2];
const names = (await readdir(directory)).filter((name) => name.endsWith('.json')).sort();
assert.deepEqual(names, ['45-sglang.json', '45-vllm.json', 'k2-sglang.json', 'k2-vllm.json']);

for (const name of names) {
  const fixture = JSON.parse(await readFile(`${directory}/${name}`, 'utf8'));
  for (const stream of [false, true]) {
    const requests = [];
    const used = [];
    const lookup = tool({
      name: 'lookup',
      description: 'Synthetic lookup',
      inputSchema: z.object({
        text: z.string(), count: z.number(), ratio: z.number(), enabled: z.boolean(),
        nested: z.record(z.string(), z.unknown()), values: z.array(z.unknown()),
      }),
      callback: (args) => { used.push(args); return 'synthetic result'; },
    });
    const fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init.body)));
      if (requests.length === 1) {
        const payload = stream
          ? fixture.stream.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
          : JSON.stringify(fixture.nonstream);
        return new Response(payload);
      }
      const final = stream
        ? `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'done' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`
        : JSON.stringify({ choices: [{ index: 0, message: { content: 'done' }, finish_reason: 'stop' }] });
      return new Response(final);
    };
    const model = new ExaoneModel({ apiKey: 'EMPTY', modelId: 'synthetic', baseURL: 'http://server.example/v1', stream, fetch });
    const agent = new Agent({ model, tools: [lookup], printer: false, retryStrategy: null });
    try {
      await agent.invoke('Use lookup', { limits: { turns: 3 } });
    } catch (error) {
      throw Error(`${name} stream=${stream}`, { cause: error });
    }
    assert.deepEqual(used, [fixture.expected]);
    assert.equal(requests.length, 2);
    const assistant = requests[1].messages.find((message) => message.role === 'assistant');
    assert.equal(assistant.content, 'synthetic answer');
    assert.equal(assistant.reasoning_content, 'synthetic reasoning');
    assert(requests[1].messages.some((message) => message.tool_call_id === 'call_0' && message.content === 'synthetic result'));
  }
  console.log(`${name}: Strands TypeScript SDK tool/reasoning round trip, stream + nonstream OK`);
}
