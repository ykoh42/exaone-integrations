import { ModelError } from '@strands-agents/sdk'
import type { ModelStreamEvent } from '@strands-agents/sdk'

/** Render a parser position in Unicode characters of the original argument text. */
export function recoveryJsonDiagnostic(error: unknown, raw: string): string {
  if (!(error instanceof SyntaxError)) return 'Invalid JSON'
  const endOfInput = error.message === 'Unexpected end of JSON input'
  const match = / at position (\d+)(?: \(line \d+ column \d+\))?$/u.exec(error.message)
  if (!match && !endOfInput) return 'Invalid JSON'
  const offset = endOfInput ? raw.length : Number(match?.[1])
  if (!Number.isSafeInteger(offset) || offset > raw.length) return 'Invalid JSON'
  const lines = raw.slice(0, offset).split('\n')
  const column = Array.from(lines.at(-1) ?? '').length + 1
  const detail = match ? error.message.slice(0, match.index) : error.message
  return `${detail} (line ${lines.length}, column ${column})`
}

/** Internal failure observed before any model event has been exposed. */
export class InvalidToolArgumentsError extends ModelError {
  #events: ModelStreamEvent[] = []
  #usage: Record<string, unknown> | undefined
  readonly #callId: string
  readonly #toolName: string
  readonly #rawArguments: string
  readonly #diagnostic: string
  constructor(callId: string, toolName: string, rawArguments: string, diagnostic: string) {
    super('EXAONE returned malformed tool arguments')
    this.name = 'InvalidToolArgumentsError'
    this.#callId = callId
    this.#toolName = toolName
    this.#rawArguments = rawArguments
    this.#diagnostic = diagnostic
  }
  get callId(): string { return this.#callId }
  get toolName(): string { return this.#toolName }
  get rawArguments(): string { return this.#rawArguments }
  get diagnostic(): string { return this.#diagnostic }
  get events(): ModelStreamEvent[] { return this.#events }
  set events(events: ModelStreamEvent[]) { this.#events = events }
  get usage(): Record<string, unknown> | undefined { return this.#usage }
  set usage(usage: Record<string, unknown> | undefined) { this.#usage = usage }
}
