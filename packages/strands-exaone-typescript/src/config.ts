import type { ExaoneModelConfig } from './types.js'
import { validateRetryConfig } from './retry.js'

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
  if (config.maxTokens !== undefined && (!Number.isInteger(config.maxTokens) || config.maxTokens < 0)) {
    throw new TypeError('maxTokens must be a non-negative integer')
  }
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
