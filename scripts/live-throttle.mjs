/** Retry only HTTP 429 responses before a comparison client sees their body. */
export function retryAfterMs(headers, now = Date.now()) {
  const delays = [10000]
  for (const [name, multiplier] of [['retry-after', 1000], ['retry-after-ms', 1]]) {
    const raw = headers.get(name)
    if (raw == null || raw.trim() === '') continue
    let delay = Number(raw) * multiplier
    if (Number.isNaN(delay) && name === 'retry-after') delay = Date.parse(raw) - now
    if (!Number.isNaN(delay) && delay >= 0) delays.push(delay)
  }
  return Math.max(...delays)
}

const maxTimerDelayMs = 2_147_483_647

function abortReason(signal) {
  return signal.reason ?? new DOMException('Aborted', 'AbortError')
}

function sleepWithSignal(delay, signal) {
  if (signal?.aborted) return Promise.reject(abortReason(signal))
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, delay)
    const onAbort = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      reject(abortReason(signal))
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

async function sleepFor(delay, signal, sleep) {
  // Node clamps overly large timers to 1 ms and truncates fractional delays.
  let remaining = Math.ceil(delay)
  while (remaining > 0) {
    if (signal?.aborted) throw abortReason(signal)
    const chunk = Math.min(remaining, maxTimerDelayMs)
    await sleep(chunk, signal)
    remaining -= chunk
  }
}

export function retry429Fetch(fetcher, { maxRetries = 20, sleep = sleepWithSignal } = {}) {
  return async (url, init) => {
    const signal = init?.signal
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (signal?.aborted) throw abortReason(signal)
      const response = await fetcher(url, init)
      if (response.status !== 429 || attempt === maxRetries) return response
      const delay = retryAfterMs(response.headers)
      if (!Number.isFinite(delay)) return response
      // A transport's cancellation can throw or never settle; do not let it
      // prevent cancellation of the retry wait or the next permitted attempt.
      try {
        void response.body?.cancel().catch(() => {})
      } catch {
        // The response body is no longer used.
      }
      await sleepFor(delay, signal, sleep)
    }
    throw new Error('429 retry loop exited without a response')
  }
}
