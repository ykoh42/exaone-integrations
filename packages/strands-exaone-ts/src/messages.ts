import type { ImageBlock, Message, StreamOptions, ToolResultBlock } from '@strands-agents/sdk'

type WireMessage = Record<string, unknown>

function modelContent<T extends { readonly type: string }>(blocks: readonly T[]): T[] {
  // SDK 1.25 retains unrecognized blocks in restored sessions for serialization only.
  return blocks.filter((block) => block.type !== 'unknownBlock')
}

function toolResult(block: ToolResultBlock): WireMessage {
  const content = modelContent(block.content).map((part) => {
    switch (part.type) {
      case 'textBlock': return part.text
      case 'jsonBlock': return JSON.stringify(part.json)
      default: throw new TypeError(`Unsupported EXAONE tool result content: ${part.type}`)
    }
  }).join('\n')
  return { role: 'tool', tool_call_id: block.toolUseId, content: block.status === 'error' ? `[ERROR] ${content}` : content }
}

function assistantMessage(blocks: Message['content']): WireMessage {
  const text: string[] = []
  const reasoning: string[] = []
  const toolCalls: WireMessage[] = []
  for (const block of blocks) {
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

function imagePart(block: ImageBlock): WireMessage {
  let url: string
  if (block.source.type === 'imageSourceBytes') {
    const serialized = block.toJSON().image.source
    if (!('bytes' in serialized) || typeof serialized.bytes !== 'string') throw new TypeError('Unable to encode image bytes')
    url = `data:image/${block.format};base64,${serialized.bytes}`
  } else if (block.source.type === 'imageSourceUrl') {
    url = block.source.url
  } else throw new TypeError('EXAONE image input requires bytes or a URL')
  return { type: 'image_url', image_url: { url } }
}

function userMessages(blocks: Message['content']): WireMessage[] {
  const results: WireMessage[] = []
  const content: WireMessage[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'textBlock': content.push({ type: 'text', text: block.text }); break
      case 'imageBlock': content.push(imagePart(block)); break
      case 'toolResultBlock': results.push(toolResult(block)); break
      default: throw new TypeError(`Unsupported EXAONE user content: ${block.type}`)
    }
  }
  if (!results.length && !content.length) throw new TypeError('EXAONE user messages cannot be empty')
  if (content.length) {
    const value = content.some((part) => part.type === 'image_url') ? content : content.map((part) => part.text).join('')
    results.push({ role: 'user', content: value })
  }
  return results
}

function systemMessages(system: StreamOptions['systemPrompt']): WireMessage[] {
  if (typeof system === 'string') return system.trim().length ? [{ role: 'system', content: system }] : []
  const text: string[] = []
  let hasCachePoints = false
  let hasGuardContent = false
  for (const block of system ?? []) {
    if (block.type === 'textBlock') text.push(block.text)
    else if (block.type === 'cachePointBlock') hasCachePoints = true
    else if (block.type === 'guardContentBlock') hasGuardContent = true
  }
  if (hasCachePoints) console.warn('cache points are not supported in EXAONE system prompts, ignoring cache points')
  if (hasGuardContent) console.warn('guard content is not supported in EXAONE system prompts, removing guard content block')
  return text.length ? [{ role: 'system', content: text.join('') }] : []
}

/** Convert Strands text, reasoning, and tool blocks to EXAONE messages. */
export function formatMessages(messages: Message[], options?: StreamOptions): WireMessage[] {
  const result: WireMessage[] = systemMessages(options?.systemPrompt)
  for (const message of messages) {
    const blocks = modelContent(message.content)
    if (!blocks.length && message.content.length) continue
    if (message.role === 'assistant') result.push(assistantMessage(blocks))
    else if (message.role === 'user') result.push(...userMessages(blocks))
    else throw new TypeError('Unsupported EXAONE message role')
  }
  return result
}
