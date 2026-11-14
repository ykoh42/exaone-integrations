import type { BaseMessage } from "@langchain/core/messages";

export interface ExaoneRequestMessage {
  role: string;
  content?: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id?: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  reasoning_content?: string;
  [key: string]: unknown;
}

function textFromContent(message: BaseMessage, allowReasoning: boolean): {
  text: string;
  reasoning: string;
} {
  const { content } = message;
  if (typeof content === "string") return { text: content, reasoning: "" };

  let text = "";
  let reasoning = "";
  for (const part of content) {
    if (typeof part === "string") {
      text += part;
      continue;
    }
    if (part.type === "text" && "text" in part && typeof part.text === "string") {
      text += part.text;
      continue;
    }
    if (
      allowReasoning &&
      part.type === "reasoning" &&
      "reasoning" in part &&
      typeof part.reasoning === "string"
    ) {
      reasoning += part.reasoning;
      continue;
    }
    if (allowReasoning && isSupportedToolCallBlock(message, part)) continue;
    throw new TypeError(
      `EXAONE Chat Completions accepts text content only; received '${part.type}'.`
    );
  }
  return { text, reasoning };
}

function isSupportedToolCallBlock(message: BaseMessage, part: Record<string, unknown>): boolean {
  if (part.type === "tool_call") {
    if (typeof part.name !== "string" || !isRecord(part.args) || (part.id != null && typeof part.id !== "string")) {
      throw new TypeError("EXAONE tool call blocks require a name, object arguments, and optional string id.");
    }
    return true;
  }
  return part.type === "tool_call_chunk" && isCompleteToolCallChunk(message, part);
}

function isCompleteToolCallChunk(message: BaseMessage, part: Record<string, unknown>): boolean {
  if (typeof part.id !== "string" || typeof part.name !== "string" || typeof part.args !== "string") return false;
  let args: unknown;
  try {
    args = JSON.parse(part.args);
  } catch {
    return false;
  }
  if (!isRecord(args)) return false;
  const calls = (message as BaseMessage & { tool_calls?: Array<{ id?: string; name: string; args: unknown }> }).tool_calls;
  return calls?.some((call) => call.id === part.id && call.name === part.name &&
    JSON.stringify(call.args) === JSON.stringify(args)) ?? false;
}

function formatToolCalls(message: BaseMessage): ExaoneRequestMessage["tool_calls"] {
  const aiMessage = message as BaseMessage & {
    tool_calls?: Array<{ id?: string; name: string; args: Record<string, unknown> }>;
    additional_kwargs?: Record<string, unknown>;
  };
  const raw = aiMessage.additional_kwargs?.tool_calls;
  if (Array.isArray(raw) && raw.length) return raw as ExaoneRequestMessage["tool_calls"];
  const original = message.response_metadata.tool_calls;
  if (Array.isArray(original) && original.length) return original as ExaoneRequestMessage["tool_calls"];

  const parsed = aiMessage.tool_calls;
  if (parsed?.length) {
    return parsed.map((call) => ({
      ...(call.id === undefined ? {} : { id: call.id }),
      type: "function",
      function: { name: call.name, arguments: JSON.stringify(call.args ?? {}) },
    }));
  }

  if (!Array.isArray(message.content)) return undefined;
  const blocks = message.content.filter((part) => typeof part === "object" && part.type === "tool_call");
  return blocks.map((part) => {
    if (typeof part === "string" || typeof part.name !== "string" || !isRecord(part.args)) {
      throw new TypeError("EXAONE tool call blocks require a name and object arguments.");
    }
    return {
      ...(typeof part.id === "string" ? { id: part.id } : {}),
      type: "function" as const,
      function: { name: part.name, arguments: JSON.stringify(part.args) },
    };
  });
}

/** Convert LangChain messages to the text-only EXAONE Chat Completions wire format. */
export function convertMessagesToExaone(messages: BaseMessage[]): ExaoneRequestMessage[] {
  return messages.map(formatMessage);
}

function formatMessage(message: BaseMessage): ExaoneRequestMessage {
  const type = message.getType();
  const { text, reasoning } = textFromContent(message, type === "ai");
  const result: ExaoneRequestMessage = {
    role: roleFor(message, type),
    content: text,
  };
  if (!result.role) throw new TypeError(`Unsupported LangChain message type '${type}'.`);
  if (message.name !== undefined) result.name = message.name;
  if (type === "tool") addToolMessageFields(message, result);
  if (type === "ai") addAssistantMessageFields(message, result, reasoning);
  return result;
}

function roleFor(message: BaseMessage, type: string): string {
  const standardRoles: Record<string, string> = {
    human: "user",
    ai: "assistant",
    system: "system",
    tool: "tool",
  };
  if (type === "chat") return String((message as BaseMessage & { role?: string }).role ?? "");
  return standardRoles[type] ?? "";
}

function addToolMessageFields(message: BaseMessage, result: ExaoneRequestMessage): void {
  const toolMessage = message as BaseMessage & { tool_call_id?: string };
  if (!toolMessage.tool_call_id) {
    throw new TypeError("LangChain ToolMessage requires a tool_call_id.");
  }
  result.tool_call_id = toolMessage.tool_call_id;
}

function addAssistantMessageFields(message: BaseMessage, result: ExaoneRequestMessage, reasoning: string): void {
  const assistantMessage = message as BaseMessage & {
    additional_kwargs?: Record<string, unknown>;
  };
  const rawReasoning = assistantMessage.additional_kwargs?.reasoning_content;
  if (reasoning) result.reasoning_content = reasoning;
  else if (typeof rawReasoning === "string") result.reasoning_content = rawReasoning;

  const legacyFunctionCall = assistantMessage.additional_kwargs?.function_call;
  if (isRecord(legacyFunctionCall)) result.function_call = legacyFunctionCall;

  const toolCalls = formatToolCalls(message);
  if (toolCalls?.length) result.tool_calls = toolCalls;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
