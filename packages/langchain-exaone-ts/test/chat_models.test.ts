import { inspect } from "node:util";
import { HumanMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it, vi, afterEach } from "vitest";
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
