import { describe, expect, it, vi } from "vitest";
import { ChatExaone } from "../src/chat_models.js";

describe("EXAONE model profiles", () => {
  it("reports known capabilities without an invented output limit", () => {
    const profile = new ChatExaone({ apiKey: "synthetic" }).profile;
    expect(profile).toMatchObject({
      maxInputTokens: 524_288, reasoningOutput: true,
      toolCalling: true, toolChoice: true,
      imageInputs: false, pdfInputs: false, audioInputs: false, videoInputs: false,
      imageUrlInputs: false, imageToolMessage: false, pdfToolMessage: false,
      imageOutputs: false, audioOutputs: false, videoOutputs: false,
    });
    expect(profile.maxOutputTokens).toBeUndefined();
    expect(profile.structuredOutput).toBeUndefined();
  });

  it("does not infer a profile for unknown models", () => {
    expect(new ChatExaone({ apiKey: "synthetic", model: "unknown-model" }).profile).toEqual({});
  });

  it.each([null, [], "profile", 1])("rejects a non-object profile: %s", (profile) => {
    expect(() => new ChatExaone({ apiKey: "synthetic", profile: profile as never }))
      .toThrow("profile must be an object");
  });

  it("uses explicit profiles as complete replacements, including empty ones", () => {
    const profile = { maxInputTokens: 4096, toolCalling: false };
    expect(new ChatExaone({ apiKey: "synthetic", profile }).profile).toEqual(profile);
    expect(new ChatExaone({ apiKey: "synthetic", profile: {} }).profile).toEqual({});
    expect(new ChatExaone({ apiKey: "synthetic", model: "unknown-model", profile }).profile).toEqual(profile);
  });

  it("does not share mutable profiles with callers or other instances", () => {
    const first = new ChatExaone({ apiKey: "synthetic" });
    first.profile.maxInputTokens = 4096;
    expect(first.profile.maxInputTokens).toBe(524_288);
    expect(new ChatExaone({ apiKey: "synthetic" }).profile.maxInputTokens).toBe(524_288);
    const profile = { fileMimeTypes: ["text/plain"] };
    const custom = new ChatExaone({ apiKey: "synthetic", profile });
    profile.fileMimeTypes.push("text/csv");
    expect(custom.profile.fileMimeTypes).toEqual(["text/plain"]);
  });

  it("does not forward profiles in API requests", async () => {
    let body: Record<string, unknown> = {};
    const fetcher = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      body = JSON.parse(String(init?.body));
      return Response.json({ choices: [{
        message: { role: "assistant", content: "Hello." }, finish_reason: "stop",
      }] });
    }) as typeof fetch;
    const model = new ChatExaone({ apiKey: "synthetic", fetch: fetcher });
    expect((await model.invoke("Hello.")).content).toBe("Hello.");
    expect(body).not.toHaveProperty("profile");
    expect(body).not.toHaveProperty("maxInputTokens");
  });
});
