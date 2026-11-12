export const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);
const MAX_TIMEOUT_MS = 2_147_483_647;

export function retryDelayMs(
  headers: Headers,
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number,
  minimumDelayMs = 0
): number | null {
  const standard = parseRetryAfter(headers.get("retry-after"));
  const milliseconds = parseRetryAfterMs(headers.get("retry-after-ms"));
  const hints = [standard, milliseconds].filter((value): value is number => value !== undefined);
  if (hints.length > 0) {
    const delay = Math.max(...hints);
    return delay <= maxDelayMs ? delay : null;
  }

  if (minimumDelayMs > maxDelayMs) return null;
  const ceiling = Math.min(maxDelayMs, baseDelayMs * 2 ** Math.min(attempt, 30));
  return Math.max(minimumDelayMs, Math.floor(Math.random() * (ceiling + 1)));
}

export function parseRetryAfter(value: string | null): number | undefined {
  if (value === null) return undefined;
  const trimmed = value.trim();
  if (trimmed === "") return undefined;
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    const seconds = Number(trimmed);
    return Number.isFinite(seconds) ? seconds * 1000 : undefined;
  }
  // Date.parse accepts non-date numeric strings such as "-1" as legacy dates.
  if (/^[+-]?\d/.test(trimmed)) return undefined;
  const timestamp = Date.parse(trimmed);
  return Number.isNaN(timestamp) ? undefined : Math.max(0, timestamp - Date.now());
}

export function parseRetryAfterMs(value: string | null): number | undefined {
  if (value === null || !/^\d+(?:\.\d+)?$/.test(value.trim())) return undefined;
  const milliseconds = Number(value);
  return Number.isFinite(milliseconds) && milliseconds >= 0 ? milliseconds : undefined;
}

function waitForChunk(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, delayMs);
    const onAbort = () => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export async function sleepWithSignal(delayMs: number, signal?: AbortSignal): Promise<void> {
  // Node clamps larger timeouts to 1 ms. Split long waits so Retry-After is not shortened.
  let remaining = Math.ceil(delayMs);
  while (remaining > MAX_TIMEOUT_MS) {
    await waitForChunk(MAX_TIMEOUT_MS, signal);
    remaining -= MAX_TIMEOUT_MS;
  }
  if (remaining > 0) await waitForChunk(remaining, signal);
}
