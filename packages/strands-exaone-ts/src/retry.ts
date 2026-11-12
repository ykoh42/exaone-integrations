import type { ExaoneModelConfig } from './types.js'

const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504])

export function validateRetryConfig(config: ExaoneModelConfig): void {
  if (config.maxRetries !== undefined && (!Number.isSafeInteger(config.maxRetries) || config.maxRetries < 0)) {
    throw new TypeError('maxRetries must be a non-negative safe integer')
  }
  const delays = [['baseRetryDelayMs', config.baseRetryDelayMs], ['maxRetryDelayMs', config.maxRetryDelayMs]] as const
  for (const [name, value] of delays) {
    if (value !== undefined && (!Number.isFinite(value) || value < 0)) {
      throw new TypeError(`${name} must be a non-negative finite number`)
    }
  }
}

function parseRetryAfter(value: string | null): number | undefined {
  if (value === null) return undefined
  const trimmed = value.trim()
  if (/^\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed) * 1000
  if (/^[+-]?\d/.test(trimmed)) return undefined
  const date = Date.parse(trimmed)
  return Number.isFinite(date) ? Math.max(0, date - Date.now()) : undefined
}

function retryAfterDelay(headers: Headers): number | undefined {
  const raw = headers.get('retry-after-ms')?.trim()
  const millis = raw && /^\d+(\.\d+)?$/.test(raw) ? Number(raw) : undefined
  const seconds = parseRetryAfter(headers.get('retry-after'))
  const delays = [millis, seconds].filter((value): value is number => value !== undefined && !Number.isNaN(value))
  return delays.length ? Math.max(...delays) : undefined
}

/** Return a bounded retry delay, or undefined when the server hint exceeds the cap. */
export function retryDelayMs(headers: Headers, attempt: number, base: number, cap: number): number | undefined {
  const requested = retryAfterDelay(headers)
  if (requested !== undefined) return requested <= cap ? requested : undefined
  const ceiling = Math.min(cap, base * 2 ** Math.min(attempt, 60))
  if (ceiling < base) return undefined
  return base + Math.random() * (ceiling - base)
}

function waitForRetryChunk(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason)
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer)
      reject(signal.reason)
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, ms)
    signal.addEventListener('abort', abort, { once: true })
  })
}

async function waitForRetry(ms: number, signal: AbortSignal): Promise<void> {
  // Node truncates fractional timeouts and clamps larger timeouts to 1 ms.
  // Round up and wait in bounded chunks so a server hint is not shortened.
  const maxTimeout = 2_147_483_647
  let remaining = Math.ceil(ms)
  while (remaining > maxTimeout) {
    await waitForRetryChunk(maxTimeout, signal)
    remaining -= maxTimeout
  }
  await waitForRetryChunk(remaining, signal)
}

/** Retry retryable HTTP responses, before exposing any events. */
export async function sendWithRetry(
  config: ExaoneModelConfig,
  signal: AbortSignal,
  send: () => Promise<Response>
): Promise<Response> {
  const retries = config.maxRetries ?? 0
  const cap = config.maxRetryDelayMs ?? 30_000
  const base = config.baseRetryDelayMs ?? 1_000
  for (let attempt = 0; ; attempt++) {
    const response = await send()
    if (!RETRYABLE_STATUSES.has(response.status) || attempt >= retries) return response
    const delay = retryDelayMs(response.headers, attempt, base, cap)
    if (delay === undefined) return response
    // Some transports reject or never resolve cancellation. Cleanup must not
    // block the retry policy or prevent the caller from cancelling the request.
    try {
      void response.body?.cancel().catch(() => undefined)
    } catch {
      // A custom transport may throw synchronously while cancelling its body.
    }
    await waitForRetry(delay, signal)
  }
}
