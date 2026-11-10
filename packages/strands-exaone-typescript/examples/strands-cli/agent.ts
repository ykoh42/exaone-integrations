import { createHarness } from '@strands-agents/harness'
import { tool } from '@strands-agents/sdk'
import { ExaoneModel } from 'strands-exaone'
import { z } from 'zod'

const getExampleNumber = tool({
  name: 'get_example_number',
  description: 'Return an example number for testing the tool loop.',
  inputSchema: z.object({}),
  callback: () => '7',
})

export const agent = await createHarness({
  model: new ExaoneModel({
    baseURL: process.env.EXAONE_BASE_URL ?? 'https://api.exaone.ai/v1',
    preserveThinking: true,
  }),
  tools: [getExampleNumber],
  builtinTools: [],
  builtinPlugins: [],
  session: false,
  memory: false,
  skills: false,
  caching: false,
})
