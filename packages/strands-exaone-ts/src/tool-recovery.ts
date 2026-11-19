import { InvokeModelStage, Message, ModelError } from '@strands-agents/sdk'
import type { AgentStreamEvent, ContentBlockData, InvokeModelResult, MiddlewareHandlerOf, ToolChoice } from '@strands-agents/sdk'
import { ExaoneAPIError, ExaoneModel } from './model.js'
import { InvalidToolArgumentsError } from './recovery-error.js'

function namedChoiceUnsupported(error: unknown): boolean {
  if (!(error instanceof ExaoneAPIError) || ![400, 422, 501].includes(error.status)) return false
  if (error.code === 'unsupported_tool_choice' || (error.param === 'tool_choice' && error.code === 'unsupported_value')) return true
  return /named (?:function|tool)|(?:function|object)[- ](?:type|valued) tool_choice/i.test(error.message) &&
    /not support|unsupported/i.test(error.message)
}

function validateCorrection({ result }: InvokeModelResult, name: string): void {
  if (result.stopReason === 'cancelled' || result.stopReason === 'maxTokens') return
  const calls = result.message.content.filter((block) => block.type === 'toolUseBlock')
  const call = calls[0]
  if (result.stopReason !== 'toolUse' || calls.length !== 1 || call?.name !== name || !call.toolUseId ||
      call.input === null || typeof call.input !== 'object' || Array.isArray(call.input)) {
    throw new ModelError('EXAONE recovery must return exactly one completed tool call for the original tool')
  }
}

async function* invokeCorrection(
  name: string,
  invoke: (choice: ToolChoice) => AsyncGenerator<AgentStreamEvent, InvokeModelResult>,
): AsyncGenerator<AgentStreamEvent, InvokeModelResult> {
  const events: AgentStreamEvent[] = []
  const collect = async (choice: ToolChoice): Promise<InvokeModelResult> => {
    const iterator = invoke(choice)
    for (let item = await iterator.next(); ; item = await iterator.next()) {
      if (item.done) return item.value
      events.push(item.value)
    }
  }
  let result: InvokeModelResult
  try { result = await collect({ tool: { name } }) } catch (error) {
    if (events.length || !namedChoiceUnsupported(error)) throw error
    result = await collect({ any: {} })
  }
  validateCorrection(result, name)
  yield* events
  return result
}

function failedCallMessages(error: InvalidToolArgumentsError): Message[] {
  let text = '', reasoning = ''
  for (const event of error.events) {
    if (event.type !== 'modelContentBlockDeltaEvent') continue
    if (event.delta.type === 'textDelta') text += event.delta.text
    if (event.delta.type === 'reasoningContentDelta') reasoning += event.delta.text ?? ''
  }
  const content: ContentBlockData[] = [
    ...(reasoning ? [{ reasoning: { text: reasoning } }] : []),
    ...(text ? [{ text }] : []),
    { toolUse: { toolUseId: error.callId, name: error.toolName, input: {} } },
  ]
  const assistant = Message.fromMessageData({ role: 'assistant', content })
  const diagnostic = JSON.stringify({ raw_arguments: error.rawArguments, parse_error: error.diagnostic })
  const feedback = 'Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n' +
    `${diagnostic}\nCall ${error.toolName} once with a valid JSON object.`
  const result = Message.fromMessageData({ role: 'user', content: [{ toolResult: { toolUseId: error.callId, status: 'error', content: [{ text: feedback }] } }] })
  return [assistant, result]
}

/**
 * Opt-in model-stage recovery for one malformed tool call before prior output.
 * Use bufferResponse when even text and reasoning must be withheld until validation.
 * Register with `agent.addMiddleware(InvokeModelStage, toolCallRecovery())`.
 */
export function toolCallRecovery({ bufferResponse = false }: { bufferResponse?: boolean } = {}): MiddlewareHandlerOf<typeof InvokeModelStage> {
  return async function* (context, next) {
    if (!(context.model instanceof ExaoneModel) || context.model.constructor !== ExaoneModel) return yield* next(context)
    const model = context.model
    const guarded = { ...context, model: model.recoveryModel(undefined, bufferResponse) }
    let exposed = false
    const iterator = next(guarded)
    try {
      while (true) {
        const item = await iterator.next()
        if (item.done) return item.value
        exposed = true
        yield item.value
      }
    } catch (error) {
      if (!(error instanceof InvalidToolArgumentsError)) throw error
      if (exposed) throw new ModelError('EXAONE returned invalid tool call arguments after output was exposed')
      const name = error.toolName
      const failedAttempt = { toolName: name, ...(error.usage === undefined ? {} : { usage: error.usage }) }
      const messages = [...context.messages, ...failedCallMessages(error)]
      return yield* invokeCorrection(name, (toolChoice) => next({
        ...guarded, messages, toolChoice, model: model.recoveryModel({ ...failedAttempt, required: 'any' in toolChoice }),
      }))
    } finally {
      await iterator.return(undefined as never)
    }
  }
}
