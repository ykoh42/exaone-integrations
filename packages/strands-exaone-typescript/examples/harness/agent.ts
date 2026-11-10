import { createHarness } from '@strands-agents/harness'
import { ExaoneModel } from 'strands-exaone'

const prompt = process.argv.slice(2).join(' ').trim()
if (!prompt) {
  console.error('Pass a prompt to the example.')
  process.exitCode = 1
} else {
  const agent = await createHarness({
    model: new ExaoneModel({ baseURL: process.env.EXAONE_BASE_URL ?? 'https://api.exaone.ai/v1' }),
    builtinTools: {
      '*': false,
      read: { media: false },
      ...(process.env.EXA_WEB_SEARCH === 'true' ? { web_search: 'exa' as const } : {}),
    },
    builtinPlugins: [],
    session: false,
    memory: false,
    skills: false,
    caching: false,
    printer: false,
  })
  try {
    const result = await agent.invoke(prompt)
    const answer = result.lastMessage.content
      .filter((block) => block.type === 'textBlock')
      .map((block) => block.text)
      .join('')
    console.log(answer)
  } finally {
    await agent.shutdown()
  }
}
