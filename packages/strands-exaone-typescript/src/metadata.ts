/** Extract provider metadata without collecting assistant content, reasoning, or tool calls. */
const RESPONSE_FIELDS = [
  'id', 'object', 'created', 'model', 'metadata', 'service_tier', 'system_fingerprint', 'moderation', 'result_code', 'description',
] as const
const CHOICE_FIELDS = ['finish_reason', 'matched_stop'] as const
const MESSAGE_FIELDS = ['refusal', 'annotations', 'audio', 'function_call'] as const

type MetadataObject = Record<string, unknown>

function isRecord(value: unknown): value is MetadataObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function copyPresentFields(target: MetadataObject, source: MetadataObject, fields: readonly string[]): void {
  for (const field of fields) {
    if (field in source && (source[field] !== null || !(field in target))) target[field] = source[field]
  }
}

function mergeLogprobs(previous: unknown, next: unknown): unknown {
  if (next === null && previous !== undefined) return previous
  if (!isRecord(previous) || !isRecord(next)) return next
  const merged = { ...previous }
  for (const [key, value] of Object.entries(next)) {
    const old = merged[key]
    if (Array.isArray(old) && Array.isArray(value)) {
      for (const item of value) old.push(item)
    } else if (value !== null || !(key in merged)) {
      merged[key] = value
    }
  }
  return merged
}

function mergeMessageValue(field: string, previous: unknown, value: unknown): unknown {
  if (field === 'refusal' && typeof previous === 'string' && typeof value === 'string') return previous + value
  if (field === 'annotations' && Array.isArray(previous) && Array.isArray(value)) {
    for (const item of value) previous.push(item)
    return previous
  }
  return value === null && previous !== undefined ? previous : value
}

function accumulateMessage(choice: MetadataObject, source: MetadataObject, streaming: boolean): void {
  const selected = MESSAGE_FIELDS.filter((field) => field in source)
  if (!selected.length) return
  const message = isRecord(choice.message) ? choice.message : {}
  if (!streaming) {
    copyPresentFields(message, source, MESSAGE_FIELDS)
    choice.message = message
    return
  }
  for (const field of ['refusal', 'annotations'] as const) {
    if (!(field in source)) continue
    message[field] = mergeMessageValue(field, message[field], source[field])
  }
  if (Object.keys(message).length) choice.message = message
  const fragment: MetadataObject = {}
  copyPresentFields(fragment, source, ['audio', 'function_call'])
  if (Object.keys(fragment).length) {
    const deltas: unknown[] = Array.isArray(choice.message_deltas) ? choice.message_deltas : []
    deltas.push(fragment)
    choice.message_deltas = deltas
  }
}

export class ResponseMetadataCollector {
  private readonly _fields: MetadataObject = {}
  private readonly _choices = new Map<number, MetadataObject>()

  accept(chunk: MetadataObject): void {
    copyPresentFields(this._fields, chunk, RESPONSE_FIELDS)
    if (!Array.isArray(chunk.choices)) return
    for (const choice of chunk.choices) {
      if (!isRecord(choice)) continue
      const index = typeof choice.index === 'number' ? choice.index : 0
      const current = this._choices.get(index) ?? { index }
      copyPresentFields(current, choice, CHOICE_FIELDS)
      if ('logprobs' in choice) current.logprobs = mergeLogprobs(current.logprobs, choice.logprobs)
      if (isRecord(choice.message)) accumulateMessage(current, choice.message, false)
      if (isRecord(choice.delta)) accumulateMessage(current, choice.delta, true)
      this._choices.set(index, current)
    }
  }

  snapshot(): MetadataObject {
    const result = { ...this._fields }
    if (this._choices.size) {
      result.choices = [...this._choices.entries()].sort(([a], [b]) => a - b).map(([, item]) => item)
    }
    return result
  }
}

/** Remove the credential from every string and map key exposed in metadata. */
export function redactMetadata(value: unknown, key: string): unknown {
  if (typeof value === 'string') return value.replaceAll(key, '[REDACTED]')
  if (Array.isArray(value)) return value.map((item) => redactMetadata(item, key))
  if (isRecord(value)) return Object.fromEntries(Object.entries(value).map(([name, item]) => [name.replaceAll(key, '[REDACTED]'), redactMetadata(item, key)]))
  return value
}
