import { Agent } from '@strands-agents/sdk'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const agent = new Agent({
  model: new ExaoneModel({ baseURL: process.env.EXAONE_BASE_URL ?? 'https://api.exaone.ai/v1' }),
  retryStrategy: new ExaoneRetryStrategy(),
  printer: false,
})

try {
  for (const prompt of ['Explain context windows in two sentences.', 'Summarize your answer in Korean.']) {
    for await (const event of agent.stream(prompt)) {
      if (event.type !== 'modelStreamUpdateEvent') continue
      const update = event.event
      if (update.type === 'modelContentBlockDeltaEvent' && update.delta.type === 'textDelta') {
        process.stdout.write(update.delta.text)
      }
    }
    console.log()
  }
} finally {
  await agent.shutdown()
}
