import type { ExaoneModelConfig } from './types.js'

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
