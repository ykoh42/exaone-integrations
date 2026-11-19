import { AIMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { InvalidToolArguments, checkToolCompletion, withRecovery } from "./recovery-context.js";
import type { RecoveryContext } from "./recovery-context.js";
import { ExaoneAPIError } from "./errors.js";

type Choice = "auto" | "none" | "required" | { type: "function"; function: { name: string } };

function namedChoiceUnsupported(error: unknown): boolean {
  if (!(error instanceof ExaoneAPIError) || ![400, 422, 501].includes(error.status)) return false;
  if (error.code === "unsupported_tool_choice" || (error.param === "tool_choice" && error.code === "unsupported_value")) return true;
  return /named (?:function|tool)|(?:function|object)[- ](?:type|valued) tool_choice/i.test(error.message) &&
    /not support|unsupported/i.test(error.message);
}

function checkModelResponse(message: AIMessage): AIMessage {
  const metadata = message.response_metadata;
  if (metadata?.model_provider === "exaone" && (message.tool_calls?.length || message.invalid_tool_calls?.length)) {
    checkToolCompletion(metadata.finish_reason, message.additional_kwargs?.refusal);
  }
  return message;
}

async function invokeCorrection<T>(name: string, retry: (choice: Choice) => Promise<T>, signal?: AbortSignal): Promise<T> {
  try {
    return await retry({ type: "function", function: { name } });
  } catch (error) {
    if (!namedChoiceUnsupported(error)) throw error;
    signal?.throwIfAborted();
    return retry("required");
  }
}

/**
 * Return configuration for LangChain's `createMiddleware`.
 *
 * A single malformed tool call is retried once before prior output is exposed.
 * Use bufferResponse to withhold the entire response until validation.
 */
export function toolCallRecovery({ bufferResponse = false }: { bufferResponse?: boolean } = {}) {
  return {
    name: "ExaoneToolCallRecovery",
    async wrapModelCall<T extends {
      messages: BaseMessage[]; model: unknown; runtime?: { signal?: AbortSignal }; toolChoice?: Choice; tools?: readonly unknown[];
    }>(
      request: T,
      handler: (request: T) => AIMessage | Promise<AIMessage>,
    ): Promise<AIMessage> {
      try {
        return checkModelResponse(await withRecovery({ bufferResponse }, async () => handler(request)));
      } catch (error) {
        if (!(error instanceof InvalidToolArguments)) throw error;
        request.runtime?.signal?.throwIfAborted();
        const call = error.details;
        const assistant = new AIMessage({
          content: call.content,
          additional_kwargs: call.reasoning === undefined ? {} : { reasoning_content: call.reasoning },
          tool_calls: [{ id: call.id, name: call.name, args: {} }],
        });
        const diagnostic = JSON.stringify({ raw_arguments: call.arguments, parse_error: call.diagnostic });
        const result = new ToolMessage({
          tool_call_id: call.id,
          status: "error",
          content: "Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n" +
            `${diagnostic}\nCall ${call.name} once with a valid JSON object.`,
        });
        const retry = async (toolChoice: Choice) => {
          const state: RecoveryContext = {
            expectedTool: call.name, required: toolChoice === "required",
            ...(call.usage === undefined ? {} : { failedUsage: call.usage }),
          };
          const response = await withRecovery(state, async () =>
            handler({ ...request, toolChoice, messages: [...request.messages, assistant, result] }));
          return { response: checkModelResponse(response), validated: state.validated };
        };
        const { response, validated } = await invokeCorrection(call.name, retry, request.runtime?.signal);
        // ToolStrategy may consume the validated call and return its own final message.
        if (validated && !response.tool_calls?.length && !response.invalid_tool_calls?.length) return response;
        if (response.response_metadata?.model_provider !== "exaone") {
          throw new Error("EXAONE recovery requires a response from an EXAONE model.");
        }
        if (response.invalid_tool_calls?.length || response.tool_calls?.length !== 1 || response.tool_calls[0]?.name !== call.name) {
          throw new Error("EXAONE recovery did not return the original tool call.");
        }
        response.response_metadata = {
          ...response.response_metadata,
          exaone_tool_call_recovery: { attempts: 1, ...(call.usage === undefined ? {} : { failed_usage: call.usage }) },
        };
        return response;
      }
    },
  };
}
