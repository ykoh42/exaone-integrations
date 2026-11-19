import { AsyncLocalStorage } from "node:async_hooks";

/** An internal, invocation-local switch. No value is sent to the server. */
export interface RecoveryContext {
  expectedTool?: string;
  bufferResponse?: boolean;
  required?: boolean;
  failedUsage?: Record<string, unknown>;
  validated?: boolean;
}

const recovery = new AsyncLocalStorage<RecoveryContext>();

export function recoveryForCall(): RecoveryContext | undefined {
  return recovery.getStore();
}

export function withRecovery<T>(options: RecoveryContext, operation: () => Promise<T>): Promise<T> {
  return recovery.run(options, operation);
}

export function isToolCompletion(finish: unknown, refusal: unknown): boolean {
  return !refusal && (finish === "tool_calls" || finish === "stop");
}

export function checkToolCompletion(finish: unknown, refusal: unknown): void {
  if (finish === "length") throw new Error("EXAONE tool calls cannot be used after a length stop.");
  if (!isToolCompletion(finish, refusal)) {
    throw new Error("EXAONE returned tool calls without a completed tool response.");
  }
}

/** Render a parser position in Unicode characters of the original argument text. */
export function recoveryJsonDiagnostic(error: unknown, raw: string): string {
  if (!(error instanceof SyntaxError)) return "Invalid JSON";
  const endOfInput = error.message === "Unexpected end of JSON input";
  const match = / at position (\d+)(?: \(line \d+ column \d+\))?$/u.exec(error.message);
  if (!match && !endOfInput) return "Invalid JSON";
  const offset = endOfInput ? raw.length : Number(match?.[1]);
  if (!Number.isSafeInteger(offset) || offset > raw.length) return "Invalid JSON";
  const lines = raw.slice(0, offset).split("\n");
  const column = Array.from(lines.at(-1) ?? "").length + 1;
  const detail = match ? error.message.slice(0, match.index) : error.message;
  return `${detail} (line ${lines.length}, column ${column})`;
}

export interface InvalidCallDetails {
  id: string;
  name: string;
  arguments: string;
  diagnostic: string;
  content: string;
  reasoning?: string;
  usage?: Record<string, unknown>;
}

/** The raw arguments are deliberately not enumerable in logged exceptions. */
export class InvalidToolArguments extends Error {
  readonly details!: InvalidCallDetails;

  constructor(details: InvalidCallDetails) {
    super("EXAONE returned malformed tool arguments");
    this.name = "InvalidToolArguments";
    Object.defineProperty(this, "details", { value: details, enumerable: false });
  }
}
