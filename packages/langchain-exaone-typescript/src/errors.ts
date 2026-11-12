import {
  parseRetryAfter as parseRetryAfterHeader,
  parseRetryAfterMs as parseRetryAfterMsHeader,
} from "./retry.js";

export type ExaoneErrorBody = {
  error?: {
    type?: string | null;
    code?: string | null;
    param?: string | null;
    message?: string;
  };
  result_code?: string | null;
  description?: string;
};

export class ExaoneAPIError extends Error {
  readonly status: number;
  readonly headers: Record<string, string>;
  readonly code: string | undefined;
  readonly errorType: string | undefined;
  readonly param: string | null | undefined;
  readonly retryAfterMs: number | undefined;

  constructor(status: number, headers: Record<string, string>, body: unknown, apiKey: string) {
    const safeBody = redactSecret(body, apiKey);
    const safeHeaders = redactSecret(headers, apiKey) as Record<string, string>;
    const envelope = isRecord(safeBody) ? safeBody : {};
    const error = isRecord(envelope.error) ? envelope.error : {};
    const code = asString(error.code) ?? asString(envelope.result_code);
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

function asStringOrNull(value: unknown): string | null | undefined {
  return value === null ? null : asString(value);
}

function redactSecret(value: unknown, secret: string): unknown {
  if (typeof value === "string") return value.split(secret).join("[REDACTED]");
  if (Array.isArray(value)) return value.map((item) => redactSecret(item, secret));
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, redactSecret(item, secret)]));
  }
  return value;
}
