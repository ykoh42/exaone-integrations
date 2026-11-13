import {
  parseRetryAfter as parseRetryAfterHeader,
  parseRetryAfterMs as parseRetryAfterMsHeader,
} from "./retry.js";

export type ExaoneErrorBody = {
  error?: {
    type?: string | null;
    code?: string | number | null;
    param?: string | null;
    message?: string;
  };
  result_code?: string | number | null;
  description?: string;
};

export class ExaoneAPIError extends Error {
  readonly status: number;
  readonly headers: Record<string, string>;
  readonly code: string | number | undefined;
  readonly errorType: string | undefined;
  readonly param: string | null | undefined;
  readonly retryAfterMs: number | undefined;

  constructor(status: number, headers: Record<string, string>, body: unknown, apiKey: string) {
    const safeBody = redactSecret(body, apiKey);
    const safeHeaders = redactSecret(headers, apiKey) as Record<string, string>;
    const envelope = isRecord(safeBody) ? safeBody : {};
    const error = isRecord(envelope.error) ? envelope.error : {};
    const code = asErrorCode(error.code) ?? asErrorCode(envelope.result_code);
    const message =
      asString(error.message) ??
      asString(envelope.description) ??
      `EXAONE API request failed with HTTP ${status}.`;
    super(message);
    this.name = "ExaoneAPIError";
    this.status = status;
    this.headers = safeHeaders;
    this.code = code;
    this.errorType = asString(error.type);
    this.param = asStringOrNull(error.param);
    const retryAfter = parseRetryAfterHeader(safeHeaders["retry-after"] ?? null);
    const retryAfterMs = parseRetryAfterMsHeader(safeHeaders["retry-after-ms"] ?? null);
    this.retryAfterMs = retryAfter === undefined && retryAfterMs === undefined
      ? undefined
      : Math.max(retryAfter ?? 0, retryAfterMs ?? 0);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function asErrorCode(value: unknown): string | number | undefined {
  if (typeof value === "string") return value.length > 0 ? value : undefined;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asStringOrNull(value: unknown): string | null | undefined {
  return value === null ? null : asString(value);
}

function redactSecret(value: unknown, secret: string): unknown {
  if (typeof value === "string") return redactSecretText(value, secret);
  if (Array.isArray(value)) return value.map((item) => redactSecret(item, secret));
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [redactSecretText(key, secret), redactSecret(item, secret)]));
  }
  return value;
}

export function safeTransportError(error: unknown, secret: string): Error {
  const original = error instanceof Error ? error : undefined;
  const message = original?.message ?? (typeof error === "string" ? error : "EXAONE transport request failed.");
  const safeMessage = redactSecretText(message, secret);
  const safe = original instanceof TypeError ? new TypeError(safeMessage) : new Error(safeMessage);
  safe.name = original?.name ? redactSecretText(original.name, secret) : "Error";
  return safe;
}

function redactSecretText(value: string, secret: string): string {
  if (!secret) return value;
  const variants = new Set([secret]);
  try {
    variants.add(encodeURIComponent(secret));
  } catch {
    // Keep the literal variant when a key contains an unpaired surrogate.
  }
  try {
    variants.add(new URLSearchParams({ value: secret }).toString().slice("value=".length));
  } catch {
    // Keep the variants that were successfully encoded.
  }
  for (const variant of [...variants].sort((left, right) => right.length - left.length)) {
    if (!variant) continue;
    value = value.split(variant).join("[REDACTED]");
    const encodedPattern = escapeRegExp(variant).replace(/%[0-9A-F]{2}/gi,
      (encoded) => encoded.replace(/[A-F]/gi, (hex) => `[${hex.toUpperCase()}${hex.toLowerCase()}]`));
    value = value.replace(new RegExp(encodedPattern, "g"), "[REDACTED]");
  }
  return value;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
