import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const agent = new Agent({
  model: new ExaoneModel({ baseURL: process.env.EXAONE_BASE_URL ?? 'https://api.exaone.ai/v1' }),
  retryStrategy: new ExaoneRetryStrategy(),
  structuredOutputSchema: z.object({ name: z.string(), email: z.string() }),
  printer: false,
})

try {
  const result = await agent.invoke('Extract the contact: Alex, alex@example.com.')
  console.log(result.structuredOutput)
} finally {
  await agent.shutdown()
}
