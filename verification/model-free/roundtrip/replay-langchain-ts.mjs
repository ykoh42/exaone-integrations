import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { ChatExaone } from 'langchain-exaone';

const directory = process.argv[2];
const names = (await readdir(directory)).filter((name) => name.endsWith('.json')).sort();
assert.deepEqual(names, ['45-sglang.json', '45-vllm.json', 'k2-sglang.json', 'k2-vllm.json']);

for (const name of names) {
  const fixture = JSON.parse(await readFile(`${directory}/${name}`, 'utf8'));
  for (const stream of [false, true]) {
    const requests = [];
    const fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init.body)));
      if (requests.length === 1) {
        const payload = stream
          ? fixture.stream.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
          : JSON.stringify(fixture.nonstream);
        return new Response(payload);
      }
      return new Response(JSON.stringify({ choices: [{ index: 0, message: { content: 'done' }, finish_reason: 'stop' }] }));
    };
    const model = new ChatExaone({ apiKey: 'EMPTY', model: 'synthetic', baseUrl: 'http://server.example/v1', outputVersion: 'v1', fetch });
    const user = new HumanMessage('Use lookup');
    let first;
    try {
      if (stream) {
        for await (const chunk of await model.stream([user])) first = first ? first.concat(chunk) : chunk;
      } else {
        first = await model.invoke([user]);
      }
      assert.equal(first.tool_calls.length, 1);
      assert.equal(first.tool_calls[0].name, 'lookup');
      assert.deepEqual(first.tool_calls[0].args, fixture.expected);
      await model.invoke([user, first, new ToolMessage({ content: 'synthetic result', tool_call_id: 'call_0' })]);
    } catch (error) {
      throw Error(`${name} stream=${stream}`, { cause: error });
    }
    assert.equal(requests.length, 2);
    const assistant = requests[1].messages.find((message) => message.role === 'assistant');
    assert.equal(assistant.content, 'synthetic answer');
    assert.equal(assistant.reasoning_content, 'synthetic reasoning');
    assert(requests[1].messages.some((message) => message.tool_call_id === 'call_0' && message.content === 'synthetic result'));
  }
  console.log(`${name}: LangChain TypeScript tool/reasoning round trip, stream + nonstream OK`);
}
