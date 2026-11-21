/** Replay pinned-source synthetic tool-choice fixtures through one TypeScript provider. */
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { createRequire } from 'node:module';

const framework = process.argv[2];
const base = new URL('./choice-fixtures/', import.meta.url);
const named = { type: 'function', function: { name: 'lookup' } };
const schema = { type: 'object', properties: { text: { type: 'string' } } };
const files = (await readdir(base)).filter((name) => name.endsWith('.json')).sort();
assert.equal(files.length, 4);

for (const name of files) {
  const fixture = JSON.parse(await readFile(new URL(name, base), 'utf8'));
  const choice = name.endsWith('-named.json') ? 'named' : 'required';
  for (const stream of [false, true]) {
    const requests = [];
    const fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init.body)));
      if (!stream) return Response.json(fixture.nonstream);
      const frames = fixture.stream.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('');
      return new Response(frames + 'data: [DONE]\n\n');
    };
    if (framework === 'strands') {
      const requireFromProvider = createRequire(import.meta.resolve('strands-exaone'));
      const { Message } = await import(requireFromProvider.resolve('@strands-agents/sdk'));
      const { ExaoneModel } = await import('strands-exaone');
      const model = new ExaoneModel({ apiKey: 'EMPTY', modelId: 'synthetic', baseURL: 'http://fixture/v1', stream, fetch });
      const user = Message.fromMessageData({ role: 'user', content: [{ text: 'lookup' }] });
      const iterator = model.streamAggregated([user], {
        toolSpecs: [{ name: 'lookup', description: 'Look up synthetic data', inputSchema: schema }],
        toolChoice: choice === 'named' ? { tool: { name: 'lookup' } } : { any: {} },
      });
      let result;
      while (true) {
        const item = await iterator.next();
        if (item.done) { result = item.value; break; }
      }
      assert.equal(result.stopReason, 'toolUse');
      const call = result.message.content.find((block) => block.type === 'toolUseBlock');
      assert.equal(call.name, 'lookup');
      assert.equal(call.toolUseId, 'call_0');
      assert.deepEqual(call.input, fixture.expected);
    } else if (framework === 'langchain') {
      const { ChatExaone } = await import('langchain-exaone');
      const model = new ChatExaone({ apiKey: 'EMPTY', model: 'synthetic', baseUrl: 'http://fixture/v1', outputVersion: 'v1', fetch });
      const bound = model.bindTools([{
        type: 'function', function: { name: 'lookup', description: 'Look up synthetic data', parameters: schema },
      }], { tool_choice: choice === 'named' ? 'lookup' : 'required' });
      let result;
      if (stream) {
        for await (const part of await bound.stream('lookup')) result = result ? result.concat(part) : part;
      } else result = await bound.invoke('lookup');
      assert.equal(result.tool_calls.length, 1);
      assert.equal(result.tool_calls[0].name, 'lookup');
      assert.equal(result.tool_calls[0].id, 'call_0');
      assert.deepEqual(result.tool_calls[0].args, fixture.expected);
    } else throw new Error(`Unknown framework: ${framework}`);
    assert.equal(requests.length, 1);
    assert.deepEqual(requests[0].tool_choice, choice === 'named' ? named : 'required');
  }
  console.log(`${name}: ${framework} streaming/nonstreaming passed`);
}
