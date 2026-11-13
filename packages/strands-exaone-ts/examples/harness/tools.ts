import { createHarness } from '@strands-agents/harness'
import { tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const getTicket = tool({
  name: 'get_ticket',
  description: 'Look up a ticket in the example support queue.',
  inputSchema: z.object({ ticket_id: z.string() }),
  callback: ({ ticket_id }) => ticket_id === 'DEMO-42'
    ? 'DEMO-42: The documentation update is ready for review. Owner: Alex.'
    : 'Ticket not found in the example queue.',
})

const agent = await createHarness({
  model: new ExaoneModel({ baseURL: process.env.EXAONE_BASE_URL ?? 'https://api.exaone.ai/v1' }),
  retryStrategy: new ExaoneRetryStrategy(),
  tools: [getTicket],
  builtinTools: { '*': false },
  builtinPlugins: [],
  session: false,
  memory: false,
  skills: false,
  caching: false,
  printer: false,
})

try {
  for (const prompt of ['Use get_ticket to check DEMO-42.', 'Who owns that ticket? Answer in Korean.']) {
    const result = await agent.invoke(prompt)
    for (const block of result.lastMessage.content) {
      if (block.type === 'textBlock') console.log(block.text)
    }
  }
} finally {
  await agent.shutdown()
}
