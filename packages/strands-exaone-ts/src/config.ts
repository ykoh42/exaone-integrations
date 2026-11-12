import type { ExaoneModelConfig } from './types.js'
import { validateRetryConfig } from './retry.js'

const MANAGED_PARAMS = new Set(['model', 'messages', 'stream', 'tools', 'tool_choice'])

function validateFiniteRange(name: string, value: number | undefined, minimum: number, maximum?: number): void {
  if (value === undefined) return
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || (maximum !== undefined && value > maximum)) {
    const range = maximum === undefined ? `at least ${minimum}` : `between ${minimum} and ${maximum}`
    throw new TypeError(`${name} must be a finite number ${range}`)
  }
}

/** Validate first-class options before storing or updating model configuration. */
export function validateModelConfig(config: ExaoneModelConfig): void {
  validateRetryConfig(config)
  validateIdentity(config)
  validateBooleanOptions(config)
  validateTimeout(config)
  validateLimits(config)
  validateModelInfo(config)
  validateReasoningEffort(config)
  validateParams(config.params)
  validateSampling(config)
}

function validateIdentity(config: ExaoneModelConfig): void {
  if (config.modelId !== undefined && (typeof config.modelId !== 'string' || config.modelId.length === 0)) {
    throw new TypeError('modelId must be a non-empty string')
  }
  if (config.baseURL !== undefined && (typeof config.baseURL !== 'string' || config.baseURL.length === 0)) {
    throw new TypeError('baseURL must be a non-empty string')
  }
  if (config.baseURL !== undefined) validateBaseURL(config.baseURL)
}

function validateBaseURL(value: string): void {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    throw new TypeError('baseURL must be an absolute URL')
  }
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) {
    throw new TypeError('baseURL must use HTTPS, except for loopback HTTP test servers')
  }
  if (url.username || url.password) throw new TypeError('baseURL must not include URL credentials')
  if (url.search || url.hash) throw new TypeError('baseURL must not include a query or fragment')
}

function validateBooleanOptions(config: ExaoneModelConfig): void {
  for (const [name, value] of Object.entries({
    stream: config.stream,
    preserveThinking: config.preserveThinking,
    enableThinking: config.enableThinking,
    includeUsage: config.includeUsage,
  })) {
    if (value !== undefined && typeof value !== 'boolean') throw new TypeError(`${name} must be a boolean`)
  }
}

function validateTimeout(config: ExaoneModelConfig): void {
  if (config.timeoutMs !== undefined && (!Number.isInteger(config.timeoutMs) || config.timeoutMs < 0 || config.timeoutMs > 4_294_967_295)) {
    throw new RangeError('timeoutMs must be an integer between 0 and 4294967295')
  }
}

function validateLimits(config: ExaoneModelConfig): void {
  if (config.maxTokens !== undefined && (!Number.isInteger(config.maxTokens) || config.maxTokens < 0)) {
    throw new TypeError('maxTokens must be a non-negative integer')
  }
  if ((config.modelId ?? 'k-exaone_v2.5') === 'k-exaone_v2.5' && (config.maxTokens ?? 0) > 524_288) {
    throw new RangeError('maxTokens cannot exceed the EXAONE context limit of 524288')
  }
  if (config.contextWindowLimit !== undefined && (!Number.isInteger(config.contextWindowLimit) || config.contextWindowLimit <= 0)) {
    throw new TypeError('contextWindowLimit must be a positive integer')
  }
}

function validateModelInfo(config: ExaoneModelConfig): void {
  for (const [field, value] of Object.entries({
    contextWindow: config.modelInfo?.limits?.contextWindow,
    maxOutputTokens: config.modelInfo?.limits?.maxOutputTokens,
  })) {
    if (value !== undefined && (!Number.isInteger(value) || value <= 0)) {
      throw new TypeError(`modelInfo.limits.${field} must be a positive integer`)
    }
  }
}

function validateReasoningEffort(config: ExaoneModelConfig): void {
  if (config.reasoningEffort !== undefined && !['low', 'medium', 'high', 'max'].includes(config.reasoningEffort)) {
    throw new TypeError('reasoningEffort must be one of low, medium, high, or max')
  }
}

function validateParams(params: ExaoneModelConfig['params']): void {
  if (params === undefined) return
  if (params === null || typeof params !== 'object' || Array.isArray(params)) {
    throw new TypeError('params must be an object')
  }
  const reserved = [...MANAGED_PARAMS].filter((name) => name in params)
  if (reserved.length) throw new TypeError(`params cannot override managed fields: ${reserved.join(', ')}`)
  if (params.n !== undefined && params.n !== 1) throw new TypeError('Strands supports exactly one completion choice')
  const template = params.chat_template_kwargs
  if (template !== undefined && (typeof template !== 'object' || template === null || Array.isArray(template))) {
    throw new TypeError('chat_template_kwargs must be an object')
  }
  const streamOptions = params.stream_options
  if (streamOptions !== undefined && streamOptions !== null && (typeof streamOptions !== 'object' || Array.isArray(streamOptions))) {
    throw new TypeError('stream_options must be an object or null')
  }
}

function validateSampling(config: ExaoneModelConfig): void {
  validateFiniteRange('temperature', config.temperature, 0)
  validateFiniteRange('topP', config.topP, 0, 1)
  validateFiniteRange('presencePenalty', config.presencePenalty, -2, 2)
  if (config.topK !== undefined && (!Number.isInteger(config.topK) || (config.topK !== -1 && config.topK < 1))) {
    throw new TypeError('topK must be -1 or a positive integer')
  }
}

function emptyCopy(value: object): object | undefined {
  const prototype = Object.getPrototypeOf(value)
  if (Array.isArray(value)) {
    if (prototype !== Array.prototype && prototype !== null) return undefined
    const copy: unknown[] = []
    if (prototype === null) Object.setPrototypeOf(copy, null)
    return copy
  }
  if (value instanceof Date && prototype === Date.prototype) return new Date(value.getTime())
  return prototype === Object.prototype || prototype === null ? Object.create(prototype) as object : undefined
}

/** Copy JSON-style configuration data without changing custom objects or callbacks. */
function copyValue(value: unknown, seen: WeakMap<object, object>): unknown {
  if (value === null || typeof value !== 'object') return value
  if (seen.has(value)) return seen.get(value)
  const copy = emptyCopy(value)
  if (!copy) return value
  seen.set(value, copy)
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!
    if ('value' in descriptor) descriptor.value = copyValue(descriptor.value, seen)
    Object.defineProperty(copy, key, descriptor)
  }
  return copy
}

/** Snapshot nested data while preserving callable and custom transport identities. */
export function copyConfig(config: ExaoneModelConfig): ExaoneModelConfig {
  return copyValue(config, new WeakMap()) as ExaoneModelConfig
}
