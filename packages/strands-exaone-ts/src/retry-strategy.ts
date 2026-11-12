import { ModelNetworkError, ModelRetryStrategy, ModelThrottledError } from '@strands-agents/sdk'
import type { AfterModelCallEvent, RetryDecision } from '@strands-agents/sdk'
import { ExaoneAPIError } from './model.js'
import { retryDelayMs } from './retry.js'

const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504])

/** Options for the EXAONE-aware Strands retry strategy. */
export interface ExaoneRetryStrategyOptions {
  /** Total attempts, including the first request. Defaults to 3. */
  maxAttempts?: number
  /** Initial delay for errors without a usable server hint. Defaults to 1000 ms. */
  baseRetryDelayMs?: number
  /** Maximum delay. A longer Retry-After hint is surfaced without retrying early. Defaults to 30000 ms. */
  maxRetryDelayMs?: number
}

function findApiError(error: unknown): ExaoneAPIError | undefined {
  const seen = new Set<Error>()
  let current = error
  while (current instanceof Error && !seen.has(current)) {
    if (current instanceof ExaoneAPIError) return current
    seen.add(current)
    current = current.cause
  }
  return undefined
}

/**
 * Strands retry strategy that honors EXAONE Retry-After headers.
 *
 * Use this in place of Strands' default strategy when the provider's rate-limit
 * guidance should control retries. Keep provider-level `maxRetries` at its
 * default of zero to avoid nested retry budgets.
 */
export class ExaoneRetryStrategy extends ModelRetryStrategy {
  readonly name = 'strands-exaone:retry-strategy'
  private readonly maxAttempts: number
  private readonly baseRetryDelayMs: number
  private readonly maxRetryDelayMs: number

  constructor(options: ExaoneRetryStrategyOptions = {}) {
    super()
    this.maxAttempts = options.maxAttempts ?? 3
    this.baseRetryDelayMs = options.baseRetryDelayMs ?? 1_000
    this.maxRetryDelayMs = options.maxRetryDelayMs ?? 30_000
    if (!Number.isSafeInteger(this.maxAttempts) || this.maxAttempts < 1) {
      throw new RangeError('maxAttempts must be a positive safe integer.')
    }
    for (const [name, value] of [
      ['baseRetryDelayMs', this.baseRetryDelayMs],
      ['maxRetryDelayMs', this.maxRetryDelayMs],
    ] as const) {
      if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be a non-negative finite number.`)
    }
  }

  protected computeRetryDecision(event: AfterModelCallEvent): RetryDecision {
    if (event.attemptCount >= this.maxAttempts) return { retry: false }
    const apiError = findApiError(event.error)
    const retryableApiError = apiError !== undefined && RETRYABLE_STATUSES.has(apiError.status)
    if (!retryableApiError && !(event.error instanceof ModelNetworkError) && !(event.error instanceof ModelThrottledError)) {
      return { retry: false }
    }

    const headers = new Headers(apiError?.headers)
    const delay = retryDelayMs(headers, event.attemptCount - 1, this.baseRetryDelayMs, this.maxRetryDelayMs)
    return delay === undefined ? { retry: false } : { retry: true, waitMs: delay }
  }
}
