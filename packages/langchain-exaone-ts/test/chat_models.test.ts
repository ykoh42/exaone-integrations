import { inspect } from "node:util";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { load } from "@langchain/core/load";
import type { AIMessageChunk } from "@langchain/core/messages";
import type { ChatGeneration } from "@langchain/core/outputs";
import { describe, expect, it, vi, afterEach } from "vitest";
import { z } from "zod";
import { ChatExaone } from "../src/chat_models.js";
import { ExaoneAPIError } from "../src/errors.js";
import { sleepWithSignal } from "../src/retry.js";

const completion = (message: Record<string, unknown>, usage?: Record<string, unknown>) =>
  new Response(
    JSON.stringify({
      id: "chatcmpl-fixture",
      model: "k-exaone_v2.5",
      choices: [{ index: 0, finish_reason: "stop", message }],
      ...(usage === undefined ? {} : { usage }),
    }),
    { headers: { "content-type": "application/json", "x-request-id": "request-fixture" } }
  );

function mockFetch(responses: Response[]) {
  const fetcher = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
    const response = responses.shift();
    if (!response) throw new Error("No fixture response remains.");
    return response;
  });
  return fetcher as unknown as typeof fetch;
}

function sseResponse(frames: unknown[]): Response {
  const encoded = new TextEncoder().encode(
    `${frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("")}data: [DONE]\n\n`
  );
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      // Split frames across reads to exercise incremental UTF-8/SSE parsing.
      for (let offset = 0; offset < encoded.length; offset += 11) {
        controller.enqueue(encoded.slice(offset, offset + 11));
      }
      controller.close();
    },
  });
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

function requestBody(fetcher: typeof fetch): Record<string, unknown> {
  const init = vi.mocked(fetcher).mock.calls.at(-1)?.[1] as RequestInit | undefined;
  if (typeof init?.body !== "string") throw new Error("Expected a JSON request body.");
  return JSON.parse(init.body) as Record<string, unknown>;
}

describe("ChatExaone", () => {
  afterEach(() => vi.useRealTimers());

  it("accepts a v1 assistant message with tool calls in content blocks", async () => {
    const fetcher = mockFetch([completion({ role: "assistant", content: "Done" })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const assistant = new AIMessage({ content: [
      { type: "reasoning", reasoning: "Need a lookup" },
      { type: "tool_call", id: "call-1", name: "lookup", args: { city: "Seoul" } },
    ], additional_kwargs: { reasoning_content: "old reasoning" } });

    await model.invoke([assistant]);

    expect(requestBody(fetcher).messages).toEqual([{
      role: "assistant",
      content: "",
      reasoning_content: "Need a lookup",
      tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: '{"city":"Seoul"}' } }],
    }]);
  });

  it("uses content-only tool calls when EXAONE metadata has no original calls", async () => {
    const fetcher = mockFetch([completion({ role: "assistant", content: "Done" })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const assistant = new AIMessage({
      content: [{ type: "tool_call", id: "call-1", name: "lookup", args: { key: "value" } }],
      response_metadata: { model_provider: "exaone" },
    });

    await model.invoke([assistant]);
    expect(requestBody(fetcher).messages).toMatchObject([{
      role: "assistant", tool_calls: [{ id: "call-1", function: { name: "lookup", arguments: '{"key":"value"}' } }],
    }]);
  });

  it("uses edited LangChain tool calls when continuing a conversation", async () => {
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "", tool_calls: [
        { id: "call-1", type: "function", provider_extra: { tag: "fixture" }, function: { name: "lookup", arguments: '{"key":"old"}', extra: "fixture" } },
      ] }),
      completion({ role: "assistant", content: "Done" }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const first = await model.invoke("Use lookup.");
    first.tool_calls![0]!.args = { key: "new" };
    await model.invoke([first]);

    const messages = requestBody(fetcher).messages as Array<{ tool_calls: Array<{ function: { arguments: string } }> }>;
    expect(JSON.parse(messages[0]!.tool_calls[0]!.function.arguments)).toEqual({ key: "new" });
    expect(messages[0]!.tool_calls[0]).toMatchObject({ provider_extra: { tag: "fixture" }, function: { extra: "fixture" } });
  });

  it("respects removed tool calls and retains invalid calls in history", async () => {
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "", tool_calls: [
        { id: "a", type: "function", function: { name: "lookup", arguments: '{"key":"a"}' } },
        { id: "b", type: "function", function: { name: "lookup", arguments: '{"key":"b"}' } },
        { id: "bad", type: "function", function: { name: "lookup", arguments: '{"key":' } },
      ] }),
      completion({ role: "assistant", content: "Done" }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const first = await model.invoke("Use lookup.");
    first.tool_calls = first.tool_calls!.slice(0, 1);
    await model.invoke([first]);

    expect(requestBody(fetcher).messages).toMatchObject([{
      role: "assistant",
      tool_calls: [{ id: "a" }, { id: "bad", function: { arguments: '{"key":' } }],
    }]);
  });

  it.each(["v0", "v1"] as const)("does not restore removed %s tool calls", async (outputVersion) => {
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "", tool_calls: [
        { id: "a", type: "function", function: { name: "lookup", arguments: "{}" } },
      ] }),
      completion({ role: "assistant", content: "Done" }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion });
    const first = await model.invoke("Use lookup.");
    first.tool_calls = [];
    await model.invoke([first]);

    expect(requestBody(fetcher).messages).toEqual([{ role: "assistant", content: "" }]);
  });

  it.each(["v0", "v1"] as const)("preserves malformed tool arguments with %s output", async (outputVersion) => {
    const rawCalls = [
      { id: "valid", type: "function", function: { name: "lookup", arguments: '{"key":"x"}' } },
      { id: "invalid", type: "function", function: { name: "lookup", arguments: '{"key":' } },
    ];
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "", tool_calls: rawCalls }),
      completion({ role: "assistant", content: "Done" }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion });

    const first = await model.invoke("Use the tools.");
    expect(first.tool_calls).toMatchObject([{ id: "valid", name: "lookup", args: { key: "x" } }]);
    expect(first.invalid_tool_calls).toMatchObject([{ id: "invalid", name: "lookup", args: '{"key":' }]);
    await model.invoke([first]);
    expect(requestBody(fetcher).messages).toMatchObject([{ role: "assistant", tool_calls: rawCalls }]);
  });

  it.each(["null", "false", "[]", '"value"', "1"])(
    "treats non-object JSON tool arguments %s as an invalid tool call",
    async (args) => {
      const fetcher = mockFetch([completion({
        role: "assistant", content: "",
        tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: args } }],
      })]);
      const result = await new ChatExaone({ apiKey: "fixture-key", fetch: fetcher }).invoke("Use a tool.");

      expect(result.tool_calls).toEqual([]);
      expect(result.invalid_tool_calls).toMatchObject([{ id: "call-1", name: "lookup", args }]);
    }
  );

  it("preserves an invalid v1 tool call without issuing an unrelated upgrade warning", async () => {
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const fetcher = mockFetch([
        completion({ role: "assistant", content: "", tool_calls: [
          { id: "call-1", type: "function", function: { name: "lookup", arguments: "null" } },
        ] }),
        completion({ role: "assistant", content: "Done" }),
      ]);
      const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion: "v1" });
      const first = await model.invoke("Use a tool.");
      expect(first.invalid_tool_calls).toMatchObject([{ id: "call-1", args: "null" }]);
      await model.invoke([first]);
      expect(requestBody(fetcher).messages).toMatchObject([{
        role: "assistant", tool_calls: [{ id: "call-1", function: { arguments: "null" } }],
      }]);
      expect(warning).not.toHaveBeenCalled();
    } finally {
      warning.mockRestore();
    }
  });

  it("exposes reasoning and tool calls in v1 non-streaming content", async () => {
    const fetcher = mockFetch([completion({
      role: "assistant",
      content: "Answer",
      reasoning_content: "Plan",
      tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: "{}" } }],
    })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion: "v1" });

    const result = await model.invoke("Question");

    expect(result.content).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "reasoning", reasoning: "Plan" }),
      expect.objectContaining({ type: "text", text: "Answer" }),
      expect.objectContaining({ type: "tool_call", id: "call-1", name: "lookup", args: {} }),
    ]));
    const restored = await load<AIMessageChunk>(JSON.stringify(result));
    expect(restored.content).toEqual(result.content);
    expect(restored.additional_kwargs).toEqual(result.additional_kwargs);
  });

  it("exposes streamed reasoning in v1 content", async () => {
    const fetcher = mockFetch([sseResponse([
      { choices: [{ index: 0, delta: { reasoning_content: "Plan" } }] },
      { choices: [{ index: 0, delta: { content: "Answer" }, finish_reason: "stop" }] },
    ])]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion: "v1" });

    const chunks = [];
    for await (const chunk of await model.stream("Question")) chunks.push(chunk);

    expect(chunks.flatMap((chunk) => Array.isArray(chunk.content) ? chunk.content : [])).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "reasoning", reasoning: "Plan" })])
    );
  });

  it("combines v1 streamed reasoning and tool-call chunks", async () => {
    const fetcher = mockFetch([sseResponse([
      { choices: [{ index: 0, delta: { reasoning_content: "Think " } }] },
      { choices: [{ index: 0, delta: { reasoning_content: "more" } }] },
      { choices: [{ index: 0, delta: { content: "Look " } }] },
      { choices: [{ index: 0, delta: { content: "up" } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call-1", function: { name: "lookup", arguments: '{"key":' } }] } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"x"}' } }] }, finish_reason: "tool_calls" }] },
    ]), completion({ role: "assistant", content: "Done" }), completion({ role: "assistant", content: "Done" })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion: "v1" });

    const chunks = [];
    for await (const chunk of await model.stream("Question")) chunks.push(chunk);
    const result = chunks.reduce((combined, chunk) => combined.concat(chunk));

    expect(result.tool_calls).toEqual([{ id: "call-1", name: "lookup", args: { key: "x" }, type: "tool_call" }]);
    expect(result.content).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: "reasoning", reasoning: "Think more" }),
      expect.objectContaining({ type: "text", text: "Look up" }),
      expect.objectContaining({ type: "tool_call_chunk", id: "call-1", name: "lookup", args: '{"key":"x"}' }),
    ]));
    result.tool_calls![0]!.args = { key: "new" };
    await model.invoke([result, new ToolMessage({ content: "Done", tool_call_id: "call-1" })]);
    expect(requestBody(fetcher).messages).toMatchObject([
      { role: "assistant", reasoning_content: "Think more", tool_calls: [{ id: "call-1", function: { arguments: '{"key":"new"}' } }] },
      { role: "tool", tool_call_id: "call-1" },
    ]);
    result.tool_calls = [];
    await model.invoke([result]);
    expect(requestBody(fetcher).messages).toMatchObject([{ role: "assistant", content: "Look up" }]);
    expect((requestBody(fetcher).messages as Array<Record<string, unknown>>)[0]).not.toHaveProperty("tool_calls");
  });

  it("rejects an incomplete v1 tool-call chunk in conversation history", async () => {
    const fetcher = mockFetch([]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const partial = new AIMessage({ content: [
      { type: "tool_call_chunk", id: "call-1", name: "lookup", args: '{"key":' },
    ] });

    await expect(model.invoke([partial])).rejects.toThrow("tool_call_chunk");
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("preserves multiple non-streaming candidates through LangChain generate", async () => {
    const response = new Response(JSON.stringify({
      model: "custom-model",
      choices: [
        { index: 0, finish_reason: "stop", message: { role: "assistant", content: "First", reasoning_content: "Plan A" } },
        { index: 1, finish_reason: "length", message: { role: "assistant", content: "Second", reasoning_content: "Plan B" } },
      ],
      usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
    }), { headers: { "content-type": "application/json" } });
    const fetcher = mockFetch([response]);
    const model = new ChatExaone({ apiKey: "fixture-key", model: "custom-model", fetch: fetcher, modelKwargs: { n: 2 } });

    const result = await model.generate([[new HumanMessage("Question")]]);
    const generations = result.generations[0] as Array<ChatGeneration & { message: AIMessageChunk }>;

    expect(requestBody(fetcher)).toMatchObject({ n: 2, stream: false });
    expect(generations.map((generation) => generation.message.content)).toEqual(["First", "Second"]);
    expect(generations.map((generation) => generation.message.additional_kwargs.reasoning_content)).toEqual(["Plan A", "Plan B"]);
    expect(generations.map((generation) => generation.generationInfo?.finish_reason)).toEqual(["stop", "length"]);
    expect(generations.map((generation) => generation.message.usage_metadata)).toEqual([
      { input_tokens: 3, output_tokens: 5, total_tokens: 8 },
      { input_tokens: 3, output_tokens: 5, total_tokens: 8 },
    ]);
    expect(result.llmOutput).toMatchObject({ tokenUsage: { promptTokens: 3, completionTokens: 5, totalTokens: 8 } });
  });

  it("rejects multiple streamed candidates instead of mixing them", async () => {
    const fetcher = mockFetch([sseResponse([{
      choices: [
        { index: 0, delta: { content: "First" } },
        { index: 1, delta: { content: "Second" } },
      ],
    }])]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, modelKwargs: { n: 2 } });

    await expect(async () => {
      for await (const _chunk of await model.stream("Question")) { /* consume the stream */ }
    }).rejects.toThrow(/multiple streamed choices/);
    expect(requestBody(fetcher)).toMatchObject({ n: 2, stream: true });
  });

  it("combines request usage across prompts without counting candidates twice", async () => {
    const usage = { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 };
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "First" }, usage),
      completion({ role: "assistant", content: "Second" }, usage),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });

    const result = await model.generate([[new HumanMessage("One")], [new HumanMessage("Two")]]);

    expect(result.generations).toHaveLength(2);
    expect(result.llmOutput).toMatchObject({ tokenUsage: { promptTokens: 6, completionTokens: 10, totalTokens: 16 } });
  });

  it("rejects a malformed candidate instead of dropping it", async () => {
    const response = new Response(JSON.stringify({
      choices: [{ index: 0, message: { role: "assistant", content: "First" } }, null],
    }));
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: mockFetch([response]) });

    await expect(model.generate([[new HumanMessage("Question")]])).rejects.toThrow(/without an assistant message/);
  });

  it.each([0, -1, 1.5, "2", true])("rejects invalid candidate count %s before sending", async (n) => {
    const fetcher = mockFetch([]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, modelKwargs: { n } });

    await expect(model.invoke("Question")).rejects.toThrow(/n must be a positive integer/);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it.each([
    "http://api.example/v1",
    "ftp://api.example/v1",
    "https://user:password@api.example/v1",
    "https://api.example/v1?token=secret",
    "https://api.example/v1#fragment",
  ])("rejects unsafe baseUrl values: %s", (baseUrl) => {
    expect(() => new ChatExaone({ apiKey: "fixture-key", baseUrl })).toThrow(/baseUrl/);
  });

  it.each(["localhost", "127.0.0.1", "[::1]"])("allows loopback HTTP for local test servers: %s", (host) => {
    const model = new ChatExaone({ apiKey: "fixture-key", baseUrl: `http://${host}:8000/v1` });
    expect(model.baseUrl).toBe(`http://${host}:8000/v1`);
  });

  it("keeps the API key out of model inspection and API errors", async () => {
    const apiKey = "synthetic-secret-marker";
    const fetcher = vi.fn(async () => new Response(
      JSON.stringify({ error: { type: "invalid_request_error", code: apiKey, message: `Rejected ${apiKey}` } }),
      { status: 400, headers: { "content-type": "application/json", "x-request-id": apiKey } }
    )) as unknown as typeof fetch;
    const model = new ChatExaone({ apiKey, fetch: fetcher });

    expect(inspect(model)).not.toContain(apiKey);
    expect(JSON.stringify(model)).not.toContain(apiKey);
    await expect(model.invoke("hello")).rejects.toMatchObject({
      message: "Rejected [REDACTED]",
      code: "[REDACTED]",
      headers: { "x-request-id": "[REDACTED]" },
    });
  });

  it("redacts URL-encoded API keys from API errors and fetch failures", async () => {
    const apiKey = "synthetic+private/key with spaces";
    const encodedKey = encodeURIComponent(apiKey);
    const formEncodedKey = new URLSearchParams({ value: apiKey }).toString().slice("value=".length);
    const differentValue = encodedKey.replace("%2F", "%3F");
    const errorFetcher = vi.fn(async () => new Response(
      JSON.stringify({ error: { message: `Rejected ${encodedKey.replace("%2F", "%2f")}`, param: differentValue } }),
      { status: 422, headers: { "x-request-id": formEncodedKey } }
    )) as unknown as typeof fetch;
    const apiError: unknown = await new ChatExaone({ apiKey, fetch: errorFetcher, maxRetries: 0 })
      .invoke("hello").catch((error: unknown) => error);

    expect(apiError).toBeInstanceOf(ExaoneAPIError);
    if (!(apiError instanceof ExaoneAPIError)) throw new Error("Expected an EXAONE API error.");
    expect(apiError.message).toBe("Rejected [REDACTED]");
    expect(apiError.headers["x-request-id"]).toBe("[REDACTED]");
    expect(apiError.param).toBe(differentValue);

    const failure = new TypeError(`Transport rejected ${formEncodedKey}`, { cause: new Error(apiKey) });
    const failingFetcher = vi.fn(async () => { throw failure; }) as unknown as typeof fetch;
    const transportError: unknown = await new ChatExaone({ apiKey, fetch: failingFetcher, maxRetries: 0 })
      .invoke("hello").catch((error: unknown) => error);

    expect(transportError).toBeInstanceOf(TypeError);
    if (!(transportError instanceof Error)) throw new Error("Expected a transport error.");
    expect(transportError.name).toBe("TypeError");
    expect(transportError.message).toBe("Transport rejected [REDACTED]");
    expect(transportError.cause).toBeUndefined();
    expect(String(transportError)).not.toContain(apiKey);
    expect(String(transportError)).not.toContain(encodedKey);
    expect(String(transportError)).not.toContain(formEncodedKey);
    expect(transportError.stack ?? "").not.toContain(apiKey);
    expect(transportError.stack ?? "").not.toContain(encodedKey);
    expect(transportError.stack ?? "").not.toContain(formEncodedKey);
  });

  it.each(["response", "stream frame", "stream tail"])("does not retain malformed JSON from a %s", async (kind) => {
    const apiKey = "synthetic+private/key";
    const malformed = `{"private":"${encodeURIComponent(apiKey)}"`;
    const body = kind === "response" ? malformed : `data: ${malformed}${kind === "stream frame" ? "\n\n" : ""}`;
    const model = new ChatExaone({ apiKey, fetch: mockFetch([new Response(body)]), maxRetries: 0 });
    let failure: unknown;
    try {
      if (kind === "response") await model.invoke("hello");
      else for await (const _chunk of await model.stream("hello")) { /* consume the response */ }
    } catch (error) { failure = error; }
    expect(failure).toBeInstanceOf(Error);
    if (!(failure instanceof Error)) throw new Error("Expected a parser error.");
    expect(failure.message).toBe("EXAONE returned malformed JSON.");
    expect(failure.cause).toBeUndefined();
    expect(failure.stack).not.toContain(encodeURIComponent(apiKey));
  });

  it.each(["response", "error response", "stream"])("sanitizes transport failures while reading a %s", async (kind) => {
    const apiKey = "synthetic+private/key";
    const response = new Response(new ReadableStream({
      start(controller) {
        controller.error(new TypeError(`Read failed ${encodeURIComponent(apiKey)}`, { cause: new Error(apiKey) }));
      },
    }), { status: kind === "error response" ? 422 : 200 });
    const model = new ChatExaone({ apiKey, fetch: mockFetch([response]), maxRetries: 0 });
    let failure: unknown;
    try {
      if (kind !== "stream") await model.invoke("hello");
      else for await (const _chunk of await model.stream("hello")) { /* consume the response */ }
    } catch (error) { failure = error; }
    expect(failure).toBeInstanceOf(Error);
    if (!(failure instanceof Error)) throw new Error("Expected a transport error.");
    expect(failure.name).toBe("TypeError");
    expect(failure).toBeInstanceOf(TypeError);
    expect(failure.message).toBe("Read failed [REDACTED]");
    expect(failure.cause).toBeUndefined();
    expect(failure.stack).not.toContain(encodeURIComponent(apiKey));
  });

  it("sends EXAONE generation settings and preserves reasoning for a follow-up turn", async () => {
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "Answer", reasoning_content: "Plan" }),
      completion({ role: "assistant", content: "Follow-up" }),
    ]);
    const model = new ChatExaone({
      apiKey: "fixture-key",
      fetch: fetcher,
      temperature: 1,
      topP: 0.95,
      topK: 30,
      reasoningEffort: "medium",
      enableThinking: true,
    });

    const first = await model.invoke("Question");
    expect(first.content).toBe("Answer");
    expect(first.additional_kwargs.reasoning_content).toBe("Plan");

    await model.invoke([new HumanMessage("Question"), first, new HumanMessage("Continue")]);
    const body = requestBody(fetcher);
    expect(body).toMatchObject({
      temperature: 1,
      top_p: 0.95,
      top_k: 30,
      reasoning_effort: "medium",
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: true },
    });
    expect(body.messages).toEqual([
      { role: "user", content: "Question" },
      { role: "assistant", content: "Answer", reasoning_content: "Plan" },
      { role: "user", content: "Continue" },
    ]);
  });

  it("preserves thinking by default and allows disabling it", async () => {
    const defaultFetcher = mockFetch([completion({ role: "assistant", content: "Answer" })]);
    await new ChatExaone({ apiKey: "fixture-key", fetch: defaultFetcher }).invoke("Question");
    expect(requestBody(defaultFetcher).chat_template_kwargs).toEqual({ preserve_thinking: true });

    const disabledFetcher = mockFetch([completion({ role: "assistant", content: "Answer" })]);
    await new ChatExaone({
      apiKey: "fixture-key",
      fetch: disabledFetcher,
      preserveThinking: false,
      modelKwargs: { chat_template_kwargs: { enable_thinking: false, preserve_thinking: true } },
    }).invoke("Question");
    expect(requestBody(disabledFetcher).chat_template_kwargs).toEqual({
      enable_thinking: false,
      preserve_thinking: false,
    });
  });

  it("uses a refusal when ordinary content is whitespace only and preserves refusal metadata", async () => {
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "\n\n", refusal: "I cannot help with that request." }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });

    const result = await model.invoke("Request");

    expect(result.content).toBe("I cannot help with that request.");
    expect(result.additional_kwargs).toMatchObject({ refusal: "I cannot help with that request." });
  });

  it("round-trips tool calls and tool results", async () => {
    const fetcher = mockFetch([
      completion({
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: "{\"key\":\"x\"}" } }],
      }),
      completion({ role: "assistant", content: "Done" }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const toolResponse = await model.invoke("Look this up.");

    expect(toolResponse.tool_calls).toEqual([{ id: "call-1", name: "lookup", args: { key: "x" } }]);
    await model.invoke([
      new HumanMessage("Look this up."),
      toolResponse,
      new ToolMessage({ content: "value", tool_call_id: "call-1" }),
    ]);

    expect(requestBody(fetcher).messages).toEqual([
      { role: "user", content: "Look this up." },
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: "{\"key\":\"x\"}" } }],
      },
      { role: "tool", content: "value", tool_call_id: "call-1" },
    ]);
  });

  it.each([
    ["non-array tool_calls", { tool_calls: { id: "call-1" } }, "EXAONE response tool_calls must be an array."],
    ["non-object tool-call entry", { tool_calls: [null] }, "EXAONE response tool call entries must be objects."],
    ["non-object tool-call function", { tool_calls: [{ function: "lookup" }] }, "EXAONE response tool call function must be an object."],
  ])("rejects malformed response %s", async (_case, message, expectedError) => {
    const fetcher = mockFetch([completion({ role: "assistant", content: "", ...message })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });

    await expect(model.invoke("Use the tool.")).rejects.toThrow(expectedError);
  });

  it.each([
    ["non-array tool_calls", { tool_calls: { index: 0 } }, "EXAONE stream delta tool_calls must be an array."],
    ["non-object tool-call entry", { tool_calls: [null] }, "EXAONE stream tool call entries must be objects."],
    ["non-object tool-call function", { tool_calls: [{ index: 0, function: "lookup" }] }, "EXAONE stream tool call function must be an object."],
  ])("rejects malformed streamed %s", async (_case, delta, expectedError) => {
    const fetcher = mockFetch([
      sseResponse([{ choices: [{ index: 0, delta }] }]),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });

    await expect(async () => {
      const stream = await model.stream("Use the tool.");
      for await (const _chunk of stream) {
        // Consume the stream so the malformed delta is parsed.
      }
    }).rejects.toThrow(expectedError);
  });

  it("retains the legacy function_call field when a conversation continues", async () => {
    const fetcher = mockFetch([
      completion({ role: "assistant", content: "", function_call: { name: "lookup", arguments: "{\"key\":\"x\"}" } }),
      completion({ role: "assistant", content: "Done" }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const first = await model.invoke("Look this up.");

    expect(first.additional_kwargs.function_call).toEqual({ name: "lookup", arguments: "{\"key\":\"x\"}" });
    await model.invoke([new HumanMessage("Look this up."), first]);

    expect(requestBody(fetcher).messages).toEqual([
      { role: "user", content: "Look this up." },
      {
        role: "assistant",
        content: "",
        function_call: { name: "lookup", arguments: "{\"key\":\"x\"}" },
      },
    ]);
  });

  it("binds LangChain tools and returns usage metadata", async () => {
    const fetcher = mockFetch([
      completion(
        { role: "assistant", content: "Done" },
        { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12, completion_tokens_details: { reasoning_tokens: 3 } }
      ),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const boundModel = model.bindTools([
      { type: "function", function: { name: "lookup", description: "Find a value", parameters: { type: "object", properties: {} } } },
    ]);

    const result = await boundModel.invoke("Use lookup.");

    expect(requestBody(fetcher).tools).toEqual([
      { type: "function", function: { name: "lookup", description: "Find a value", parameters: { type: "object", properties: {} } } },
    ]);
    expect(result.usage_metadata).toEqual({
      input_tokens: 5,
      output_tokens: 7,
      total_tokens: 12,
      output_token_details: { reasoning: 3 },
    });
    expect(result.response_metadata).toMatchObject({ model_provider: "exaone", model_name: "k-exaone_v2.5" });
    expect(result.response_metadata.headers).toMatchObject({ "x-request-id": "request-fixture" });
  });

  it("preserves top-level reasoning usage when completion details omit it", async () => {
    const fetcher = mockFetch([
      completion(
        { role: "assistant", content: "Done" },
        { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12, completion_tokens_details: {}, reasoning_tokens: 3 }
      ),
    ]);
    const result = await new ChatExaone({ apiKey: "fixture-key", fetch: fetcher }).invoke("Hello");

    expect(result.usage_metadata?.output_token_details).toEqual({ reasoning: 3 });
  });

  it("supports LangChain structured output", async () => {
    const fetcher = mockFetch([
      completion({
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call-structured", type: "function", function: { name: "extract", arguments: '{"answer":"42"}' } }],
      }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const structured = model.withStructuredOutput(
      { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
      { name: "extract" }
    );

    await expect(structured.invoke("Return an answer.")).resolves.toEqual({ answer: "42" });
    expect(requestBody(fetcher).tools).toMatchObject([
      { type: "function", function: { name: "extract" } },
    ]);
    expect(requestBody(fetcher).tool_choice).toEqual({ type: "function", function: { name: "extract" } });
  });

  it.each(["jsonMode", "jsonSchema"] as const)("rejects unsupported structured-output method %s", (method) => {
    const model = new ChatExaone({ apiKey: "fixture-key" });
    expect(() => model.withStructuredOutput(z.object({ answer: z.string() }), { method })).toThrow("functionCalling");
  });

  it("supports structured output with a Zod schema and the raw message", async () => {
    const fetcher = mockFetch([completion({
      role: "assistant", content: "", tool_calls: [
        { id: "call-1", type: "function", function: { name: "extract", arguments: '{"answer":"42"}' } },
      ],
    })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const structured = model.withStructuredOutput(z.object({ answer: z.string() }), { includeRaw: true });

    const result = await structured.invoke("Return an answer.");
    expect(result.parsed).toEqual({ answer: "42" });
    expect(AIMessage.isInstance(result.raw)).toBe(true);
  });

  it("keeps the raw response when structured-output parsing fails", async () => {
    const fetcher = mockFetch([completion({ role: "assistant", content: "No tool call" })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const structured = model.withStructuredOutput(z.object({ answer: z.string() }), { includeRaw: true });

    const result = await structured.invoke("Return an answer.");
    expect(result.parsed).toBeNull();
    expect(AIMessage.isInstance(result.raw)).toBe(true);
  });

  it("validates Zod structured output through LangChain's parser", async () => {
    const fetcher = mockFetch([completion({
      role: "assistant", content: "", tool_calls: [
        { id: "call-1", type: "function", function: { name: "extract", arguments: '{"answer":42}' } },
      ],
    })]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const structured = model.withStructuredOutput(z.object({ answer: z.string() }), { includeRaw: true });

    const result = await structured.invoke("Return an answer.");
    expect(result.parsed).toBeNull();
    expect(AIMessage.isInstance(result.raw) && result.raw.tool_calls?.[0]?.args).toEqual({ answer: 42 });
  });

  it("streams structured output through LangChain's tool parser", async () => {
    const fetcher = mockFetch([sseResponse([
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call-1", function: { name: "extract", arguments: '{"answer":' } }] } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"42"}' } }] }, finish_reason: "tool_calls" }] },
    ])]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const structured = model.withStructuredOutput(z.object({ answer: z.string() }));

    const results = [];
    for await (const item of await structured.stream("Return an answer.")) results.push(item);
    expect(results.at(-1)).toEqual({ answer: "42" });
  });

  it("accumulates split SSE frames including reasoning, tool-call chunks, and usage", async () => {
    const fetcher = mockFetch([
      sseResponse([
        { id: "stream-1", choices: [{ index: 0, delta: { reasoning_content: "Think" } }] },
        { id: "stream-1", choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call-1", function: { name: "lookup", arguments: "{\"key\":" } }] } }] },
        { id: "stream-1", choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: "\"x\"}" } }] }, finish_reason: "tool_calls" }] },
        { id: "stream-1", choices: [], usage: { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 } },
      ]),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const chunks = [];
    for await (const chunk of await model.stream("Use lookup.")) chunks.push(chunk);
    const final = chunks.reduce((message, chunk) => message.concat(chunk));

    expect(chunks.map((chunk) => chunk.additional_kwargs.reasoning_content).filter(Boolean)).toEqual(["Think"]);
    expect(final.tool_calls).toEqual([{ id: "call-1", name: "lookup", args: { key: "x" }, type: "tool_call" }]);
    expect(final.usage_metadata).toMatchObject({ input_tokens: 4, output_tokens: 6, total_tokens: 10 });
  });

  it("exposes reasoning through LangChain's typed streamEvents API", async () => {
    const fetcher = mockFetch([
      sseResponse([
        { choices: [{ index: 0, delta: { reasoning_content: "Think " } }] },
        { choices: [{ index: 0, delta: { content: "Answer " } }] },
        { choices: [{ index: 0, delta: { reasoning_content: "more" } }] },
        { choices: [{ index: 0, delta: { content: "done" } }] },
      ]),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const stream = model.streamEvents("Explain this.");

    await expect(stream.reasoning).resolves.toBe("Think more");
    await expect(stream.text).resolves.toBe("Answer done");
    const message = await stream;
    expect(message.content).toEqual([
      { type: "text", text: "Answer done", index: 0 },
      { type: "reasoning", reasoning: "Think more", index: 1 },
    ]);
  });

  it("supports typed stream events with v1 output", async () => {
    const fetcher = mockFetch([sseResponse([
      { choices: [{ index: 0, delta: { reasoning_content: "Think" } }] },
      { choices: [{ index: 0, delta: { content: "Answer" } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call-1", function: { name: "lookup", arguments: "{}" } }] }, finish_reason: "tool_calls" }] },
    ])]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, outputVersion: "v1" });
    const stream = model.streamEvents("Question");

    await expect(stream.reasoning).resolves.toBe("Think");
    await expect(stream.text).resolves.toBe("Answer");
    const message = await stream;
    expect(message.tool_calls).toEqual([{ id: "call-1", name: "lookup", args: {}, type: "tool_call" }]);
  });

  it("uses a streamed refusal when the corresponding content is whitespace only", async () => {
    const fetcher = mockFetch([
      sseResponse([
        { choices: [{ index: 0, delta: { content: "\n\n", refusal: "I cannot help with that request." } }] },
      ]),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher });
    const chunks = [];
    for await (const chunk of await model.stream("Request")) chunks.push(chunk);
    const final = chunks.reduce((message, chunk) => message.concat(chunk));

    expect(final.content).toBe("I cannot help with that request.");
    expect(final.additional_kwargs.refusal).toBe("I cannot help with that request.");
  });

  it("honors Retry-After without a short default timeout", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ error: { message: "slow down" } }), {
        status: 429,
        headers: { "content-type": "application/json", "retry-after": "1" },
      }))
      .mockResolvedValueOnce(completion({ role: "assistant", content: "Recovered" }));
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher as unknown as typeof fetch, maxRetries: 1 });

    const pending = model.invoke("Retry safely.");
    await vi.advanceTimersByTimeAsync(999);
    expect(fetcher).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect((await pending).content).toBe("Recovered");
    expect(fetcher).toHaveBeenCalledTimes(2);
    const firstInit = fetcher.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(firstInit?.signal).toBeUndefined();
  });

  it("waits at least one second before retrying a 429 with no retry hint", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ error: { message: "service busy" } }), {
        status: 429,
        headers: { "content-type": "application/json" },
      }))
      .mockResolvedValueOnce(completion({ role: "assistant", content: "Recovered" }));
    const model = new ChatExaone({
      apiKey: "fixture-key",
      fetch: fetcher as unknown as typeof fetch,
      maxRetries: 1,
      baseRetryDelayMs: 100,
      maxRetryDelayMs: 2_000,
    });

    const pending = model.invoke("Retry safely.");
    await vi.advanceTimersByTimeAsync(999);
    expect(fetcher).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect((await pending).content).toBe("Recovered");
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("treats a negative Retry-After value as invalid before applying the 429 minimum wait", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ error: { message: "service busy" } }), {
        status: 429,
        headers: { "content-type": "application/json", "retry-after": "-1" },
      }))
      .mockResolvedValueOnce(completion({ role: "assistant", content: "Recovered" }));
    const model = new ChatExaone({
      apiKey: "fixture-key",
      fetch: fetcher as unknown as typeof fetch,
      maxRetries: 1,
      baseRetryDelayMs: 100,
      maxRetryDelayMs: 2_000,
    });

    const pending = model.invoke("Retry safely.");
    await vi.advanceTimersByTimeAsync(999);
    expect(fetcher).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(pending).resolves.toMatchObject({ content: "Recovered" });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("waits in Node-safe chunks when Retry-After exceeds the timer limit", async () => {
    vi.useFakeTimers();
    const maximumTimeout = 2_147_483_647;
    const timeoutSpy = vi.spyOn(globalThis, "setTimeout");
    const pending = sleepWithSignal(maximumTimeout + 1);

    try {
      await vi.advanceTimersByTimeAsync(maximumTimeout);
      expect(timeoutSpy.mock.calls.map(([, delay]) => delay)).toEqual([maximumTimeout, 1]);
      await vi.advanceTimersByTimeAsync(1);
      await expect(pending).resolves.toBeUndefined();
    } finally {
      timeoutSpy.mockRestore();
    }
  });

  it("retries a transient HTTP 408 response", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ error: { message: "request timed out" } }), {
        status: 408,
        headers: { "content-type": "application/json" },
      }))
      .mockResolvedValueOnce(completion({ role: "assistant", content: "Recovered" }));
    const model = new ChatExaone({
      apiKey: "fixture-key",
      fetch: fetcher as unknown as typeof fetch,
      maxRetries: 1,
      baseRetryDelayMs: 1,
      maxRetryDelayMs: 1,
    });

    const pending = model.invoke("Retry safely.");
    await vi.advanceTimersByTimeAsync(1);
    await expect(pending).resolves.toMatchObject({ content: "Recovered" });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("does not retry a hintless 429 when the retry cap is under one second", async () => {
    const fetcher = mockFetch([
      new Response(JSON.stringify({ error: { message: "service busy" } }), { status: 429 }),
      completion({ role: "assistant", content: "Should not be requested" }),
    ]);
    const model = new ChatExaone({
      apiKey: "fixture-key",
      fetch: fetcher,
      maxRetries: 1,
      baseRetryDelayMs: 100,
      maxRetryDelayMs: 500,
    });

    await expect(model.invoke("Retry safely.")).rejects.toMatchObject({ status: 429 });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("exposes API error status and typed envelope fields", async () => {
    const fetcher = mockFetch([
      new Response(JSON.stringify({ error: { type: "invalid_request_error", code: "bad_param", param: "top_k", message: "Invalid top_k." } }), {
        status: 422,
        headers: { "content-type": "application/json", "retry-after-ms": "250" },
      }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, maxRetries: 0 });

    await expect(model.invoke("Request")).rejects.toMatchObject({
      name: "ExaoneAPIError",
      status: 422,
      code: "bad_param",
      errorType: "invalid_request_error",
      param: "top_k",
      retryAfterMs: 250,
    } satisfies Partial<ExaoneAPIError>);
  });

  it("preserves numeric EXAONE result codes", async () => {
    const fetcher = mockFetch([
      new Response(JSON.stringify({ result_code: 500, description: "The service is temporarily unavailable." }), {
        status: 500,
        headers: { "content-type": "application/json" },
      }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, maxRetries: 0 });

    await expect(model.invoke("Request")).rejects.toMatchObject({
      name: "ExaoneAPIError",
      status: 500,
      code: 500,
      message: "The service is temporarily unavailable.",
    } satisfies Partial<ExaoneAPIError>);
  });

  it("falls back to a top-level result code when a nested error omits its code", async () => {
    const fetcher = mockFetch([
      new Response(JSON.stringify({ error: { message: "The service is temporarily unavailable." }, result_code: 10042 }), {
        status: 500,
        headers: { "content-type": "application/json" },
      }),
    ]);
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, maxRetries: 0 });

    await expect(model.invoke("Request")).rejects.toMatchObject({
      name: "ExaoneAPIError",
      status: 500,
      code: 10042,
      message: "The service is temporarily unavailable.",
    } satisfies Partial<ExaoneAPIError>);
  });

  it("rejects unsupported multimodal input instead of silently dropping it", async () => {
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: mockFetch([]) });
    const multimodal = new HumanMessage({ content: [{ type: "image_url", image_url: "https://example.test/image.png" }] });

    await expect(model.invoke([multimodal])).rejects.toThrow("accepts text content only");
  });

  it("does not set a timeout unless requested", () => {
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: mockFetch([]) });
    expect(model.timeout).toBeUndefined();
  });

  it("stops waiting for a stream when the caller aborts", async () => {
    const controller = new AbortController();
    const encoder = new TextEncoder();
    let streamWasCancelled = false;
    const fetcher = vi.fn(async () => new Response(new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(encoder.encode('data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'));
      },
      cancel() {
        streamWasCancelled = true;
      },
    }), { headers: { "content-type": "text/event-stream" } }));
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher as unknown as typeof fetch });
    const stream = await model.stream("Stream slowly.", { signal: controller.signal });
    const iterator = stream[Symbol.asyncIterator]();

    expect((await iterator.next()).value?.content).toBe("first");
    const pendingRead = iterator.next();
    controller.abort(new DOMException("Stopped by test", "AbortError"));

    await expect(pendingRead).rejects.toMatchObject({ name: "AbortError" });
    expect(streamWasCancelled).toBe(true);
  });
});
