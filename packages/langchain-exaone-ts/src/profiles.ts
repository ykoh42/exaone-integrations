import type { ModelProfile } from "@langchain/core/language_models/profile";

/** Model facts for the EXAONE API: https://docs.exaone.ai/ko/docs/k-exaone-2.5. @internal */
export function getModelProfile(model: string): ModelProfile {
  if (model !== "k-exaone_v2.5") return {};
  return {
    maxInputTokens: 524_288,
    imageInputs: false,
    imageUrlInputs: false,
    imageToolMessage: false,
    pdfToolMessage: false,
    pdfInputs: false,
    audioInputs: false,
    videoInputs: false,
    imageOutputs: false,
    audioOutputs: false,
    videoOutputs: false,
    reasoningOutput: true,
    toolCalling: true,
    toolChoice: true,
  };
}
