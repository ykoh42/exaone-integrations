import { HumanMessage } from '@langchain/core/messages'
import { ChatExaone } from 'langchain-exaone'

const model = new ChatExaone({
  reasoningEffort: 'medium',
  enableThinking: true,
})

const question = new HumanMessage('Explain how a tool-using agent works.')
const response = await model.invoke([question])

console.log(response.content)
console.log(response.additional_kwargs.reasoning_content)

const followUp = await model.invoke([
  question,
  response,
  new HumanMessage('Summarize the answer in one sentence.'),
])

console.log(followUp.content)
