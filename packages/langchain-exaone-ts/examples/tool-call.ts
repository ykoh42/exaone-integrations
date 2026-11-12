import { ChatExaone } from 'langchain-exaone'

const model = new ChatExaone()
const modelWithTools = model.bindTools([
  {
    type: 'function',
    function: {
      name: 'lookup_weather',
      description: 'Look up the weather for a city.',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
        additionalProperties: false,
      },
    },
  },
])

const response = await modelWithTools.invoke('What is the weather in Seoul?')
for (const call of response.tool_calls ?? []) {
  console.log(call.name, call.args)
  // Run the requested tool and pass its output back in a ToolMessage.
}
