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

function textFromContent(content: BaseMessage["content"], allowReasoning: boolean): {
  text: string;
  reasoning: string;
} {
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
    throw new TypeError(
      `EXAONE Chat Completions accepts text content only; received '${part.type}'.`
    );
  }
  return { text, reasoning };
}

function formatToolCalls(message: BaseMessage): ExaoneRequestMessage["tool_calls"] {
  const aiMessage = message as BaseMessage & {
    tool_calls?: Array<{ id?: string; name: string; args: Record<string, unknown> }>;
    additional_kwargs?: Record<string, unknown>;
  };
  const parsed = aiMessage.tool_calls;
  if (parsed?.length) {
    return parsed.map((call) => ({
      ...(call.id === undefined ? {} : { id: call.id }),
      type: "function",
      function: { name: call.name, arguments: JSON.stringify(call.args ?? {}) },
    }));
  }

  const raw = aiMessage.additional_kwargs?.tool_calls;
  return Array.isArray(raw)
    ? (raw as ExaoneRequestMessage["tool_calls"])
    : undefined;
}

/** Convert LangChain messages to the text-only EXAONE Chat Completions wire format. */
export function convertMessagesToExaone(messages: BaseMessage[]): ExaoneRequestMessage[] {
  return messages.map(formatMessage);
}

function formatMessage(message: BaseMessage): ExaoneRequestMessage {
  const type = message.getType();
  const { text, reasoning } = textFromContent(message.content, type === "ai");
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
  if (typeof rawReasoning === "string") result.reasoning_content = rawReasoning;
  else if (reasoning) result.reasoning_content = reasoning;

  const legacyFunctionCall = assistantMessage.additional_kwargs?.function_call;
  if (isRecord(legacyFunctionCall)) result.function_call = legacyFunctionCall;

  const toolCalls = formatToolCalls(message);
  if (toolCalls?.length) result.tool_calls = toolCalls;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
