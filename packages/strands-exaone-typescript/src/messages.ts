import type { Message, StreamOptions, ToolResultBlock } from '@strands-agents/sdk'

type WireMessage = Record<string, unknown>

function toolResult(block: ToolResultBlock): WireMessage {
  const content = block.content.map((part) => {
    switch (part.type) {
      case 'textBlock': return part.text
      case 'jsonBlock': return JSON.stringify(part.json)
      default: throw new TypeError(`Unsupported EXAONE tool result content: ${part.type}`)
    }
  }).join('\n')
  return { role: 'tool', tool_call_id: block.toolUseId, content: block.status === 'error' ? `[ERROR] ${content}` : content }
}

function assistantMessage(message: Message): WireMessage {
  const text: string[] = []
  const reasoning: string[] = []
  const toolCalls: WireMessage[] = []
  for (const block of message.content) {
    switch (block.type) {
      case 'textBlock': text.push(block.text); break
      case 'reasoningBlock':
        if (block.signature || block.redactedContent) throw new TypeError('EXAONE cannot preserve reasoning signatures or redacted reasoning')
        if (block.text !== undefined) reasoning.push(block.text)
        break
      case 'toolUseBlock':
        if (block.reasoningSignature) throw new TypeError('EXAONE cannot preserve tool reasoning signatures')
        toolCalls.push({ id: block.toolUseId, type: 'function', function: { name: block.name, arguments: JSON.stringify(block.input) } })
        break
      default: throw new TypeError(`Unsupported EXAONE assistant content: ${block.type}`)
    }
  }
  const result: WireMessage = { role: 'assistant', content: text.length ? text.join('') : null }
  if (reasoning.length) result.reasoning_content = reasoning.join('')
  if (toolCalls.length) result.tool_calls = toolCalls
  return result
}

function userMessages(message: Message): WireMessage[] {
  const results: WireMessage[] = []
  const text: string[] = []
  for (const block of message.content) {
    switch (block.type) {
      case 'textBlock': text.push(block.text); break
      case 'toolResultBlock': results.push(toolResult(block)); break
      default: throw new TypeError(`Unsupported EXAONE user content: ${block.type}`)
    }
  }
  if (!results.length && !text.length) throw new TypeError('EXAONE user messages cannot be empty')
  if (text.length) results.push({ role: 'user', content: text.join('') })
  return results
}

function systemMessages(system: StreamOptions['systemPrompt']): WireMessage[] {
  if (typeof system === 'string') {
    return system.length ? [{ role: 'system', content: system }] : []
  }
  const text: string[] = []
  for (const block of system ?? []) {
    if (block.type !== 'textBlock') throw new TypeError(`Unsupported EXAONE system content: ${block.type}`)
    text.push(block.text)
  }
  return text.length ? [{ role: 'system', content: text.join('') }] : []
}

/** Convert Strands blocks to the text-only EXAONE message protocol. */
export function formatMessages(messages: Message[], options?: StreamOptions): WireMessage[] {
  const result: WireMessage[] = systemMessages(options?.systemPrompt)
  for (const message of messages) {
    if (message.role === 'assistant') result.push(assistantMessage(message))
    else if (message.role === 'user') result.push(...userMessages(message))
    else throw new TypeError('Unsupported EXAONE message role')
  }
  return result
}
