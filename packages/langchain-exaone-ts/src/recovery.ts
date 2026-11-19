import { AIMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { InvalidToolArguments, checkToolCompletion, correctionPrompt, namedChoiceUnsupported, withRecovery } from "./recovery-context.js";
import type { RecoveryContext } from "./recovery-context.js";

type Choice = "auto" | "none" | "required" | { type: "function"; function: { name: string } };

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

/** Enable correction for a single malformed JSON tool call. */
export function toolCallRecovery() {
  return {
    name: "ExaoneToolCallRecovery",
    async wrapModelCall<T extends {
      messages: BaseMessage[]; model: unknown; runtime?: { signal?: AbortSignal }; toolChoice?: Choice; tools?: readonly unknown[];
    }>(
      request: T,
      handler: (request: T) => AIMessage | Promise<AIMessage>,
    ): Promise<AIMessage> {
      try {
        return checkModelResponse(await withRecovery({}, async () => handler(request)));
      } catch (error) {
        if (!(error instanceof InvalidToolArguments)) throw error;
        request.runtime?.signal?.throwIfAborted();
        const call = error.details;
        const assistant = new AIMessage({
          content: call.content,
          additional_kwargs: call.reasoning === undefined ? {} : { reasoning_content: call.reasoning },
          tool_calls: [{ id: call.id, name: call.name, args: {} }],
        });
        const result = new ToolMessage({
          tool_call_id: call.id,
          status: "error",
          content: correctionPrompt(call),
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
