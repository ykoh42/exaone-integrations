import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const multiply = tool({
  name: 'multiply',
  description: 'Multiply two integers.',
  inputSchema: z.object({ a: z.number().int(), b: z.number().int() }),
  callback: ({ a, b }) => a * b,
})

const agent = new Agent({
  model: new ExaoneModel({ baseURL: process.env.EXAONE_BASE_URL ?? 'https://api.exaone.ai/v1' }),
  retryStrategy: new ExaoneRetryStrategy(),
  tools: [multiply],
  systemPrompt: 'Use the multiply tool for multiplication.',
  printer: false,
})

try {
  const result = await agent.invoke('What is 6 times 7?')
  for (const block of result.lastMessage.content) {
    if (block.type === 'textBlock') console.log(block.text)
  }
} finally {
  await agent.shutdown()
}
