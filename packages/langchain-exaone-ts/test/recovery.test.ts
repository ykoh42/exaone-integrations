import { createAgent, createMiddleware, toolStrategy } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import { InMemoryCache } from "@langchain/core/caches";
import { tool } from "@langchain/core/tools";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { ChatExaone } from "../src/chat_models.js";
import { toolCallRecovery } from "../src/recovery.js";

type Scenario = "normal" | "multi-valid" | "invalid" | "mixed" | "mixed-valid-first" | "repeated" | "length" | "length-valid" | "length-empty" | "length-whitespace" | "split" | "repeated-id" |
  "empty" | "whitespace" | "empty-object" | "fallback" | "bad-request" | "no-tool" | "wrong-tool" | "extra-tool" | "blank" | "array" | "multiline" |
  "tool-only" | "live-text" | "duplicate-finish" | "legacy-original" | "legacy-correction";
type TestToolChoice = "auto" | "required" | { type: "function"; function: { name: string } };
const namedRecord = { type: "function", function: { name: "record" } };
const multilineArguments = '{\n"note":"😀```","amount":123,,}';

function call(id: string, args: string) {
  return { id, type: "function", function: { name: "record", arguments: args } };
}

function harness(
  scenario: Scenario, enabled = true, wrapped = false, checkpointer?: MemorySaver,
  options?: { strict: boolean; toolChoice: TestToolChoice },
  failedUsage?: Record<string, unknown>, apiKey = "fixture-key",
  emptyTool?: "zero" | "defaulted",
) {
  const requests: Array<Record<string, any>> = [];
  const executed: number[] = [];
  let releaseResponse: (() => void) | undefined;
  const fetcher: typeof fetch = async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    requests.push(body);
    const n = requests.length;
    if (n === 2 && (scenario === "fallback" || scenario === "bad-request")) {
      return Response.json({ error: { message: scenario === "fallback" ? "Named tool choice is not supported" : "invalid request to model" } }, { status: 400 });
    }
    let calls: Array<ReturnType<typeof call>> | undefined;
    const normal = scenario === "normal" || scenario === "multi-valid" || scenario === "duplicate-finish";
    if (n === 1) {
      const argumentsText = scenario === "empty" || scenario === "length-empty" ? "" : scenario === "whitespace" || scenario === "length-whitespace" ? " \n\t" :
        scenario === "empty-object" ? "{}" : normal || scenario === "length-valid" ? '{"amount":123}' :
          scenario === "multiline" ? multilineArguments : '{"amount":123,}';
      calls = [call("original", argumentsText)];
      if (scenario === "multi-valid") calls.push(call("other", '{"amount":456}'));
      if (scenario === "mixed" || scenario === "mixed-valid-first") {
        calls.push(call("other", '{"amount":7}'));
        if (scenario === "mixed-valid-first") calls.reverse();
      }
    } else if (n === (scenario === "fallback" ? 3 : 2) && !normal && !["empty", "whitespace", "empty-object", "no-tool"].includes(scenario)) {
      calls = [call("corrected", scenario === "repeated" ? '{"amount":123,}' : scenario === "blank" ? " " : scenario === "array" ? "[]" : '{"amount":123}')];
      if (scenario === "wrong-tool") calls[0]!.function.name = "other";
      if (scenario === "extra-tool") calls.push(call("extra", '{"amount":456}'));
    }
    const finish = scenario.startsWith("length") && n === 1 ? "length" : calls ? "tool_calls" : "stop";
    if (!body.stream) {
      const usage = n === 1 && failedUsage ? failedUsage : { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 };
      return Response.json({ id: "fixture", model: "fixture", usage, choices: [{ index: 0, finish_reason: finish,
        message: { role: "assistant", content: calls ? (n === 1 && scenario === "invalid" ? "failure-only-content" : null) : "done", ...(calls ? { tool_calls: calls } : {}) } }] });
    }
    if (n === 1 && scenario === "live-text") {
      const encoder = new TextEncoder();
      const frame = (delta: Record<string, unknown>, finish_reason?: string) =>
        `data: ${JSON.stringify({ id: "fixture", choices: [{ index: 0, delta, finish_reason }] })}\n\n`;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(frame({ content: "Checking.", tool_calls: [{ ...calls![0], index: 0 }] })));
          releaseResponse = () => {
            const usage = { choices: [], usage: { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 } };
            controller.enqueue(encoder.encode(`${frame({}, "tool_calls")}data: ${JSON.stringify(usage)}\n\ndata: [DONE]\n\n`));
            controller.close();
            releaseResponse = undefined;
          };
        },
        cancel() { releaseResponse = undefined; },
      });
      return new Response(stream, { headers: { "content-type": "text/event-stream" } });
    }
    const frames: unknown[] = [{ id: "fixture", choices: [{ index: 0, delta: { role: "assistant" } }] }];
    if (calls) {
      if (scenario !== "tool-only") {
        frames.push({ id: "fixture", choices: [{ index: 0, delta: { reasoning_content: n === 1 ? "failure-only-reasoning" : "corrected-reasoning" } }] });
      }
      if (n === 1 && (scenario === "split" || scenario === "repeated-id")) {
        frames.push({ id: "fixture", choices: [{ index: 0,
          delta: { tool_calls: [{ index: 0, id: "original", type: "function", function: { name: "record", arguments: '{"amount":' } }] } }] });
        frames.push({ id: "fixture", choices: [{ index: 0,
          delta: { tool_calls: [{ index: 0, ...(scenario === "repeated-id" ? { id: "original" } : {}), function: { arguments: "123,}" } }] } }] });
      } else {
        const legacy = (scenario === "legacy-original" && n === 1) || (scenario === "legacy-correction" && n === 2);
        frames.push({ id: "fixture", choices: [{ index: 0,
          delta: { ...(scenario === "live-text" && n === 2 ? { content: "Private correction preamble." } : {}),
            ...(legacy ? { content: "Synthetic text.", function_call: { name: "legacy", arguments: "{}" } } : {}),
            tool_calls: calls.map((item, index) => ({ ...item, index })) } }] });
      }
    } else {
      frames.push({ id: "fixture", choices: [{ index: 0, delta: { content: "done" } }] });
    }
    const terminal = { id: "fixture", choices: [{ index: 0, delta: {}, finish_reason: finish }] };
    if (n === 1 && scenario === "duplicate-finish") frames.push(terminal);
    frames.push(terminal);
    if (n === 1 && failedUsage) frames.push({ id: "fixture", choices: [], usage: failedUsage });
    if (n === 2 && scenario === "live-text") {
      frames.push({ choices: [], usage: { prompt_tokens: 7, completion_tokens: 8, total_tokens: 15 } });
    }
    return new Response(`${frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("")}data: [DONE]\n\n`,
      { headers: { "content-type": "text/event-stream" } });
  };
  const record = emptyTool === "zero"
    ? tool(() => { executed.push(0); return "Recorded."; }, {
      name: "record", description: "Record a synthetic amount.", schema: z.object({}),
    })
    : tool(({ amount }) => { executed.push(amount); return "Recorded."; }, {
      name: "record", description: "Record a synthetic amount.",
      schema: z.object({ amount: emptyTool === "defaulted" ? z.number().default(99) : z.number() }),
    });
  const model = new ChatExaone({ apiKey, fetch: fetcher, maxRetries: 0 });
  const settings = options && createMiddleware({
    name: "SyntheticToolSettings",
    wrapModelCall: async (request, handler) => handler({
      ...request,
      toolChoice: options.toolChoice,
      modelSettings: { ...request.modelSettings, strict: options.strict },
    }),
  });
  const agent = createAgent({ model: (wrapped ? model.withConfig({ tags: ["fixture"] }) : model) as typeof model,
    tools: [record], systemPrompt: "synthetic system",
    ...(checkpointer === undefined ? {} : { checkpointer }),
    middleware: enabled ? [...(settings ? [settings] : []), createMiddleware(toolCallRecovery())] : [] });
  return { agent, requests, executed, finishResponse: () => releaseResponse?.() };
}

describe("tool-call recovery", () => {
  it.each(["filter initial", "filter correction", "override choice", "route model", "route mixed calls"])(
    "uses the downstream model request when middleware will %s", async (scenario) => {
      const requests: Array<{tools?: unknown[]; tool_choice?: unknown}> = [];
      const executed: number[] = [];
      const fetcher: typeof fetch = async (_input, init) => {
        requests.push(JSON.parse(String(init?.body)));
        const n = requests.length;
        const toolCalls = [call(`attempt-${n}`, n === 1 ? '{"amount":123,}' : '{"amount":123}')];
        if (n === 1 && scenario === "route mixed calls") toolCalls.push(call("valid-sibling", '{"amount":123}'));
        const message = n > 2 ? { content: "Done." } : { tool_calls: toolCalls };
        return Response.json({ choices: [{ index: 0, message: { role: "assistant", ...message }, finish_reason: n > 2 ? "stop" : "tool_calls" }] });
      };
      const record = tool(({ amount }) => { executed.push(amount); return "Recorded."; }, {
        name: "record", description: "Record a synthetic amount.", schema: z.object({ amount: z.number() }),
      });
      const model = new ChatExaone({ apiKey: "fixture", fetch: fetcher, maxRetries: 0 });
      const routed = new ChatExaone({ apiKey: "fixture", fetch: fetcher, maxRetries: 0 });
      const downstream = createMiddleware({ name: "DownstreamModelPolicy", wrapModelCall: (request, handler) => {
        if (scenario.startsWith("route")) return handler({ ...request, model: routed });
        if (scenario === "filter initial" || (scenario === "filter correction" && request.toolChoice != null)) {
          return handler({ ...request, tools: [] });
        }
        if (scenario === "override choice" && request.toolChoice != null) return handler({ ...request, toolChoice: "none" });
        return handler(request);
      } });
      const agent = createAgent({ model, tools: [record], middleware: [createMiddleware(toolCallRecovery()), downstream] });
      const input = { messages: [{ role: "user" as const, content: "Record 123." }] };
      if (scenario === "route model") {
        await agent.invoke(input);
        expect(requests.map((request) => request.tool_choice)).toEqual([undefined, namedRecord, undefined]);
        expect(executed).toEqual([123]);
      } else {
        const reason = scenario === "route mixed calls" ? "multiple tool calls" : "does not allow the tool correction";
        await expect(agent.invoke(input)).rejects.toThrow(reason);
        expect(requests).toHaveLength(1);
        expect(Boolean(requests[0]?.tools?.length)).toBe(scenario !== "filter initial");
        expect(executed).toEqual([]);
      }
    },
  );

  it("returns SDK ToolStrategy results with failed usage on cold and warm correction cache", async () => {
    const choices: unknown[] = [];
    let name: string | undefined;
    const fetcher: typeof fetch = async (_input, init) => {
      const body = JSON.parse(String(init?.body));
      choices.push(body.tool_choice);
      name ??= body.tools[0].function.name;
      const initial = body.tool_choice === "required";
      const syntheticCall = initial ? call("failed", '{"value":3,}') : call("corrected", '{"value":3}');
      syntheticCall.function.name = name!;
      const usage = initial ? { prompt_tokens: choices.length, completion_tokens: 4, total_tokens: choices.length + 4 }
        : { prompt_tokens: 20, completion_tokens: 4, total_tokens: 24 };
      return Response.json({ choices: [{ index: 0, finish_reason: "tool_calls", message: { role: "assistant", tool_calls: [syntheticCall] } }], usage });
    };
    const cache = new InMemoryCache();
    // The agent includes a per-run ID in its metadata. Only the model request identifies this fixture.
    cache.makeDefaultKeyEncoder((prompt, key) => JSON.stringify([prompt, key.replace(/(^|,)executionInfo:\{[^}]*\}(?=,|$)/g, "$1").replace(/,,/g, ",")]));
    const model = new ChatExaone({ apiKey: "fixture-key", fetch: fetcher, maxRetries: 0, cache });
    const agent = createAgent({ model, tools: [], responseFormat: toolStrategy(z.object({ value: z.number() })),
      middleware: [createMiddleware(toolCallRecovery())] });
    for (const failedInputTokens of [1, 3]) {
      const response = await agent.invoke({ messages: [{ role: "user", content: "Return a synthetic amount." }] });
      expect(response.structuredResponse).toEqual({ value: 3 });
      expect(response.messages).toContainEqual(expect.objectContaining({
        response_metadata: expect.objectContaining({
          exaone_tool_call_recovery: { attempts: 1, failed_usage: { prompt_tokens: failedInputTokens, completion_tokens: 4, total_tokens: failedInputTokens + 4 } },
        }),
      }));
    }
    expect(choices).toEqual(["required", { type: "function", function: { name } }, "required"]);
  });

  it("passes one finish reason to the SDK when the server repeats it", async () => {
    const { agent, requests, executed } = harness("duplicate-finish");
    const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
    const finishReasons: unknown[] = [];
    for await (const [message] of events) finishReasons.push(message.response_metadata?.finish_reason);
    expect(finishReasons.filter((reason) => reason === "tool_calls")).toEqual(["tool_calls"]);
    expect(requests[0]?.stream).toBe(true);
    expect(requests).toHaveLength(2);
    expect(executed).toEqual([123]);
  });

  it("shares guarded cache entries without reusing results from plain model calls", async () => {
    const requests: Record<string, unknown>[] = [];
    const fetcher: typeof fetch = async (_url, init) => {
      const body = JSON.parse(String(init?.body));
      requests.push(body);
      const finish = body.messages.at(-1).content === "Truncated response." ? "length" : "tool_calls";
      return Response.json({ choices: [{ index: 0, finish_reason: finish,
        message: { role: "assistant", tool_calls: [call("original", '{"amount":123}')] } }] });
    };
    const record = tool(() => "Recorded.", {
      name: "record", description: "Record a synthetic amount.", schema: z.object({ amount: z.number() }),
    });
    const model = new ChatExaone({ apiKey: "fixture", cache: new InMemoryCache(), fetch: fetcher, maxRetries: 0 });
    const bound = model.bindTools([record]);
    const middleware = toolCallRecovery();
    const first = { model: bound, messages: [new HumanMessage("Truncated response.")], tools: [record] };
    const invoke = async (request: typeof first) => {
      const message: unknown = await bound.invoke(request.messages);
      if (!AIMessage.isInstance(message)) throw new Error("Expected an assistant message.");
      return message;
    };

    expect((await invoke(first)).tool_calls).toHaveLength(1);
    await expect(middleware.wrapModelCall(first, invoke)).rejects.toThrow("length stop");
    expect(requests).toHaveLength(2);
    expect(requests[1]).toEqual(requests[0]);

    const completed = { ...first, messages: [new HumanMessage("Completed response.")] };
    expect((await middleware.wrapModelCall(completed, invoke)).tool_calls).toHaveLength(1);
    expect((await middleware.wrapModelCall(completed, invoke)).tool_calls).toHaveLength(1);
    expect(requests).toHaveLength(3);
  });

  it.each(["missing finish", "content filter", "refusal", "conflicting finish"])(
    "holds tool events when the completion reports %s", async (scenario) => {
      let requests = 0;
      const executed: number[] = [];
      const record = tool(({ amount }) => { executed.push(amount); return "Recorded."; }, {
        name: "record", description: "Record a synthetic amount.", schema: z.object({ amount: z.number() }),
      });
      const toolCall = call("original", '{"amount":123}');
      const delta = { tool_calls: [{ ...toolCall, index: 0 }], ...(scenario === "refusal" ? { refusal: "Declined." } : {}) };
      const finish = scenario === "missing finish" ? null : scenario === "content filter" ? "content_filter" : "tool_calls";
      const fetcher: typeof fetch = async () => {
        requests++;
        const frames = [
          { choices: [{ index: 0, delta, finish_reason: scenario === "conflicting finish" ? "length" : null }] },
          { choices: [{ index: 0, delta: {}, finish_reason: finish }] },
        ];
        return new Response(frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("") + "data: [DONE]\n\n");
      };
      const model = new ChatExaone({ apiKey: "fixture", fetch: fetcher, maxRetries: 0 });
      const agent = createAgent({ model, tools: [record], middleware: [createMiddleware(toolCallRecovery())] });
      const visible: unknown[] = [];
      await expect(async () => {
        const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
        for await (const [message] of events) visible.push(message);
      }).rejects.toThrow(scenario === "conflicting finish" ? "conflicting stream finish reasons" : "without a completed tool response");
      expect(JSON.stringify(visible)).not.toContain("original");
      if (scenario === "refusal") expect(JSON.stringify(visible)).toContain("Declined.");
      expect(requests).toBe(1);
      expect(executed).toEqual([]);
    },
  );

  it("streams text immediately and completes the same message after correcting the tool", async () => {
    const { agent, requests, executed, finishResponse } = harness("live-text");
    const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] },
      { streamMode: "messages", signal: AbortSignal.timeout(4000) });
    const textIds: Array<string | undefined> = [];
    const toolIds: Array<string | undefined> = [];
    const usage: unknown[] = [];
    try {
      for await (const [message] of events) {
        expect(message.content).not.toBe("Private correction preamble.");
        if ("usage_metadata" in message && message.usage_metadata) {
          usage.push({ raw: message.response_metadata?.usage, combined: message.usage_metadata,
            correction: message.response_metadata?.exaone_tool_call_recovery });
        }
        if (message.content === "Checking.") {
          textIds.push(message.id);
          finishResponse();
        }
        if ("tool_call_chunks" in message && Array.isArray(message.tool_call_chunks) && message.tool_call_chunks.length) {
          toolIds.push(message.id);
        }
      }
    } finally {
      finishResponse();
    }
    expect(textIds).toHaveLength(1);
    expect(toolIds).toEqual(textIds);
    expect(requests).toHaveLength(3);
    expect(executed).toEqual([123]);
    expect(usage).toMatchObject([{ raw: { total_tokens: 10 }, combined: { total_tokens: 25 },
      correction: { correction_usage: { total_tokens: 15 } } }]);
  });

  it.each(["legacy-original", "legacy-correction"] as const)("blocks mixed tool calls during %s", async (scenario) => {
    const { agent, requests, executed } = harness(scenario);
    await expect(async () => {
      const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
      for await (const _event of events) { /* consume until validation fails */ }
    }).rejects.toThrow(scenario === "legacy-original" ? "unsupported assistant content" : "legacy function call");
    expect(executed).toEqual([]);
    expect(requests).toHaveLength(scenario === "legacy-original" ? 1 : 2);
  });

  it("recovers a tool-only stream without buffering ordinary text", async () => {
    const { agent, requests, executed } = harness("tool-only", true, false, undefined, undefined, undefined,
      "fixture-key", undefined);
    const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
    for await (const _event of events) { /* consume to completion */ }
    expect(requests.map((request) => request.tool_choice)).toEqual([undefined, namedRecord, undefined]);
    expect(executed).toEqual([123]);
  });

  it("retries a malformed call through a matched error tool message", async () => {
    const { agent, requests, executed } = harness("invalid");
    const state = await agent.invoke({ messages: [{ role: "user", content: "Record 123." }] });
    expect(requests).toHaveLength(3);
    expect(executed).toEqual([123]);
    for (const request of requests) {
      expect(request.messages.filter((message: any) => message.role === "system")).toEqual([
        { role: "system", content: "synthetic system" },
      ]);
      expect(request.messages.filter((message: any) => message.role === "user")).toHaveLength(1);
    }
    expect(requests.map((request) => request.tool_choice)).toEqual([undefined, namedRecord, undefined]);
    const retry = requests[1]!.messages;
    const assistant = retry.find((message: any) => message.role === "assistant");
    const result = retry.find((message: any) => message.role === "tool");
    expect(assistant.tool_calls[0]).toMatchObject({ id: "original", function: { name: "record", arguments: "{}" } });
    expect(result.tool_call_id).toBe("original");
    expect(Object.keys(result).sort()).toEqual(["content", "role", "tool_call_id"]);
    const [header, diagnostic, instruction] = String(result.content).split("\n");
    expect(header).toMatch(/^Error: .*not executed\. The previous tool call is a placeholder\.$/);
    expect(JSON.parse(diagnostic!)).toMatchObject({ raw_arguments: '{"amount":123,}', parse_error: expect.stringContaining("line 1, column 15") });
    expect(instruction).toMatch(/^Call record once/);
    expect(assistant.content).toBe("failure-only-content");
    const next = requests[2]!.messages;
    const nextCalls = next.filter((message: any) => message.role === "assistant").flatMap((message: any) => message.tool_calls ?? []);
    expect(nextCalls).toEqual([call("corrected", '{"amount":123}')]);
    expect(next.filter((message: any) => message.role === "tool").map((message: any) => message.tool_call_id)).toEqual(["corrected"]);
    expect(JSON.stringify(next)).not.toContain("failure-only-content");
    expect(JSON.stringify(state.messages)).not.toContain("failure-only-content");
    expect(JSON.stringify(state.messages)).not.toContain('"tool_call_id":"original"');
    expect(JSON.stringify(state.messages)).not.toContain('{"amount":123,}');
    const recovered = state.messages.find((message) => message.response_metadata?.exaone_tool_call_recovery);
    expect(recovered?.response_metadata.exaone_tool_call_recovery).toEqual({
      attempts: 1, failed_usage: { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 },
    });
  });

  it.each(["invoke", "stream"] as const)("redacts nested failed usage credentials and preserves cache details on %s", async (mode) => {
    const secret = "synthetic key/segment";
    const encoded = encodeURIComponent(secret);
    const formEncoded = new URLSearchParams({ value: secret }).toString().slice("value=".length);
    const lowercaseEncoded = encoded.replace("%2F", "%2f");
    const usage = {
      prompt_tokens: 40, completion_tokens: 60, total_tokens: 100,
      prompt_tokens_details: { cached_tokens: 12, cache_write_tokens: 8 },
      completion_tokens_details: { reasoning_tokens: 5 },
      [secret]: { value: secret, [encoded]: [encoded, formEncoded, `prefix ${lowercaseEncoded} suffix`] },
    };
    const { agent, executed } = harness("invalid", true, false, new MemorySaver(), undefined, usage, secret);
    const config = { configurable: { thread_id: `synthetic-failed-usage-${mode}` } };
    if (mode === "invoke") {
      await agent.invoke({ messages: [{ role: "user", content: "Record 123." }] }, config);
    } else {
      const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { ...config, streamMode: "updates" });
      for await (const _event of events) { /* consume to completion */ }
    }
    const snapshot = await agent.graph.getState(config);
    const recovered = snapshot.values.messages.find((message: any) => message.response_metadata?.exaone_tool_call_recovery);
    const failedUsage = recovered.response_metadata.exaone_tool_call_recovery.failed_usage;
    expect(failedUsage).toMatchObject({
      prompt_tokens: 40, completion_tokens: 60, total_tokens: 100,
      prompt_tokens_details: { cached_tokens: 12, cache_write_tokens: 8 },
      completion_tokens_details: { reasoning_tokens: 5 },
      "[REDACTED]": { value: "[REDACTED]", "[REDACTED]": ["[REDACTED]", "[REDACTED]", "prefix [REDACTED] suffix"] },
    });
    const serialized = JSON.stringify(failedUsage);
    for (const variant of [secret, encoded, formEncoded, lowercaseEncoded]) expect(serialized).not.toContain(variant);
    expect(executed).toEqual([123]);
  });

  it.each([
    [true, "auto"], [false, "auto"], [true, "required"], [false, "required"],
    [true, { type: "function", function: { name: "record" } }],
    [false, { type: "function", function: { name: "record" } }],
  ] as const)("preserves strict=%s and the tool schema on recovery", async (strict, toolChoice) => {
    for (const mode of ["invoke", "stream"] as const) {
      const { agent, requests, executed } = harness("invalid", true, false, undefined, { strict, toolChoice });
      const input = { messages: [{ role: "user" as const, content: "Record 123." }] };
      if (mode === "invoke") {
        await agent.invoke(input);
      } else {
        const events = await agent.stream(input, { streamMode: "updates" });
        for await (const _event of events) { /* consume to completion */ }
      }
      expect(requests).toHaveLength(3);
      expect(executed).toEqual([123]);
      expect(requests[0]!.tool_choice).toEqual(toolChoice);
      expect(requests[1]!.tool_choice).toEqual(namedRecord);
      expect(requests[2]!.tool_choice).toEqual(toolChoice);
      expect(requests[0]!.tools[0].function).toMatchObject({
        name: "record", strict,
        parameters: { type: "object", properties: { amount: { type: "number" } }, required: ["amount"] },
      });
      expect(requests[1]!.tools).toEqual(requests[0]!.tools);
      expect(requests[1]!.messages.filter((message: any) => message.role === "system"))
        .toEqual(requests[0]!.messages.filter((message: any) => message.role === "system"));
      expect(requests[1]!.messages.filter((message: any) => message.role === "user"))
        .toEqual(requests[0]!.messages.filter((message: any) => message.role === "user"));
    }
  });

  it.each(["invoke", "stream"] as const)("resumes a %s checkpoint without persisting the failed-call placeholder", async (mode) => {
    const { agent, requests, executed } = harness("invalid", true, false, new MemorySaver());
    const config = { configurable: { thread_id: "synthetic-recovery-session" } };
    let firstMessages;
    if (mode === "invoke") {
      firstMessages = (await agent.invoke({ messages: [{ role: "user", content: "Record 123." }] }, config)).messages;
    } else {
      const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { ...config, streamMode: "updates" });
      for await (const _item of stream) { /* consume to completion */ }
    }
    const checkpoint = await agent.graph.getState(config);
    const checkpointMessages = checkpoint.values.messages;
    for (const messages of [firstMessages ?? checkpointMessages, checkpointMessages]) {
      const calls = messages.flatMap((message: any) => message.tool_calls ?? []);
      expect(calls).toHaveLength(1);
      expect(calls[0]).toMatchObject({ id: "corrected", name: "record", args: { amount: 123 } });
      const results = messages.filter((message: any) => message.type === "tool");
      expect(results).toHaveLength(1);
      expect(results[0].tool_call_id).toBe("corrected");
      expect(JSON.stringify(messages)).not.toContain("failure-only-content");
      expect(JSON.stringify(messages)).not.toContain("failure-only-reasoning");
      expect(JSON.stringify(messages)).not.toContain('{"amount":123,}');
      expect(JSON.stringify(messages)).not.toContain("The previous tool call is a placeholder.");
    }
    const resumed = await agent.invoke({ messages: [{ role: "user", content: "Continue." }] }, config);
    expect(requests).toHaveLength(4);
    expect(executed).toEqual([123]);
    const resumedMessages = requests[3]!.messages;
    expect(resumedMessages.filter((message: any) => message.role === "user").map((message: any) => message.content))
      .toEqual(["Record 123.", "Continue."]);
    const calls = resumedMessages.filter((message: any) => message.role === "assistant").flatMap((message: any) => message.tool_calls ?? []);
    expect(calls).toEqual([call("corrected", '{"amount":123}')]);
    expect(resumedMessages.filter((message: any) => message.role === "tool").map((message: any) => message.tool_call_id)).toEqual(["corrected"]);
    expect(JSON.stringify(resumedMessages)).not.toContain("The previous tool call is a placeholder.");
    expect(JSON.stringify(resumedMessages)).not.toContain("failure-only-reasoning");
    expect(JSON.stringify(resumed.messages)).not.toContain("The previous tool call is a placeholder.");
  });

  it("passes normal requests through without changing their wire body", async () => {
    const plain = harness("normal", false);
    const guarded = harness("normal", true);
    await plain.agent.invoke({ messages: [{ role: "user", content: "Record 123." }] });
    await guarded.agent.invoke({ messages: [{ role: "user", content: "Record 123." }] });
    expect(guarded.requests).toEqual(plain.requests);
    expect(guarded.executed).toEqual([123]);
  });

  it.each(["empty", "whitespace", "empty-object"] as const)(
    "preserves the SDK's %s argument behavior for zero-argument and defaulted tools", async (scenario) => {
      for (const mode of ["invoke", "stream"] as const) {
        for (const variant of ["zero", "defaulted"] as const) {
          const plain = harness(scenario, false, false, undefined, undefined, undefined, "fixture-key", variant);
          const guarded = harness(scenario, true, false, undefined, undefined, undefined, "fixture-key", variant);
          const run = async (fixture: typeof plain) => {
            const input = { messages: [{ role: "user" as const, content: "Use the synthetic tool." }] };
            if (mode === "invoke") {
              const state = await fixture.agent.invoke(input);
              return state.messages.map((message: any) => ({
                type: message.type, content: message.content,
                tool_calls: message.tool_calls, invalid_tool_calls: message.invalid_tool_calls,
              }));
            }
            const stream = await fixture.agent.stream(input, { streamMode: "messages" });
            const messages = [];
            for await (const [message] of stream) {
              const item = message as AIMessage;
              messages.push({ type: message.type, content: message.content,
                tool_calls: item.tool_calls, invalid_tool_calls: item.invalid_tool_calls });
            }
            return messages;
          };
          const before = await run(plain);
          const after = await run(guarded);
          expect(guarded.requests).toEqual(plain.requests);
          expect(after).toEqual(before);
          expect(guarded.executed).toEqual(plain.executed);
          expect(guarded.requests[0]!.stream).toBe(mode === "stream");
          const sdkExecutes = mode === "stream" || scenario === "empty-object";
          expect(guarded.executed).toEqual(sdkExecutes ? [variant === "zero" ? 0 : 99] : []);
          expect(guarded.requests).toHaveLength(sdkExecutes ? 2 : 1);
        }
      }
    },
  );

  it("exposes recovery metadata in streamed agent updates", async () => {
    const { agent, executed } = harness("invalid");
    const updates: unknown[] = [];
    const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "updates" });
    for await (const update of stream) updates.push(update);
    expect(executed).toEqual([123]);
    expect(JSON.stringify(updates)).toContain("exaone_tool_call_recovery");
  });

  it("enables the guard on a model wrapped with LangChain withConfig", async () => {
    const { agent, requests, executed } = harness("invalid", true, true);
    await agent.invoke({ messages: [{ role: "user", content: "Record 123." }] });
    expect(requests).toHaveLength(3);
    expect(executed).toEqual([123]);
    expect(requests[1]!.messages.some((message: any) => message.role === "tool" && message.tool_call_id === "original"))
      .toBe(true);
  });

  it.each([false, true])("preserves multiple valid calls with optional recovery (stream=%s)", async (stream) => {
    const baseline = harness("multi-valid", false);
    const guarded = harness("multi-valid");
    const invoke = async (agent: typeof baseline.agent) => {
      if (!stream) {
        await agent.invoke({ messages: [{ role: "user", content: "Record 123." }] });
        return;
      }
      const updates = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "updates" });
      for await (const _ of updates) { /* consume the stream */ }
    };
    await invoke(baseline.agent);
    await invoke(guarded.agent);
    expect(baseline.requests).toEqual(guarded.requests);
    expect([...baseline.executed].sort()).toEqual([123, 456]);
    expect([...guarded.executed].sort()).toEqual([123, 456]);
  });

  it.each(["mixed", "mixed-valid-first"] as const)("blocks mixed streamed calls (%s)", async (scenario) => {
    const { agent, requests, executed } = harness(scenario);
    await expect(async () => {
      const updates = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "updates" });
      for await (const _ of updates) { /* consume the stream */ }
    }).rejects.toThrow();
    expect(requests).toHaveLength(1);
    expect(executed).toEqual([]);
  });

  it("does not execute sibling tools when a mixed response contains an invalid call", async () => {
    const { agent, requests, executed } = harness("mixed");
    await expect(agent.invoke({ messages: [{ role: "user", content: "Record 123." }] })).rejects.toThrow();
    expect(requests).toHaveLength(1);
    expect(executed).toEqual([]);
  });

  it("stops after one recovery attempt", async () => {
    const { agent, requests, executed } = harness("repeated");
    await expect(agent.invoke({ messages: [{ role: "user", content: "Record 123." }] })).rejects.toThrow();
    expect(requests).toHaveLength(2);
    expect(executed).toEqual([]);
  });

  it.each(["invoke", "stream"] as const)("falls back to required only when named choice is explicitly rejected (%s)", async (mode) => {
    for (const scenario of ["fallback", "bad-request"] as const) {
      const { agent, requests, executed } = harness(scenario);
      const input = { messages: [{ role: "user" as const, content: "Record 123." }] };
      const run = async () => {
        if (mode === "invoke") {
          await agent.invoke(input);
          return;
        }
        for await (const _event of await agent.stream(input, { streamMode: "messages" })) { /* consume */ }
      };
      if (scenario === "fallback") {
        await run();
        expect(requests.map((request) => request.tool_choice)).toEqual([undefined, namedRecord, "required", undefined]);
        expect(requests[1]!.messages).toEqual(requests[2]!.messages);
        expect(executed).toEqual([123]);
      } else {
        await expect(run()).rejects.toThrow();
        expect(requests.map((request) => request.tool_choice)).toEqual([undefined, namedRecord]);
        expect(executed).toEqual([]);
      }
    }
  });

  it.each(["no-tool", "wrong-tool", "extra-tool", "blank", "array"] as const)(
    "does not expose or execute an invalid correction (%s)", async (scenario) => {
      for (const mode of ["invoke", "stream"] as const) {
        const { agent, requests, executed } = harness(scenario);
        const input = { messages: [{ role: "user" as const, content: "Record 123." }] };
        const exposed: unknown[] = [];
        await expect(async () => {
          if (mode === "invoke") await agent.invoke(input);
          else for await (const event of await agent.stream(input, { streamMode: "messages" })) exposed.push(event);
        }).rejects.toThrow();
        expect(exposed.length > 0).toBe(mode === "stream");
        expect(JSON.stringify(exposed)).not.toContain("corrected-reasoning");
        expect(exposed.every((item) => !Array.isArray(item) ||
          !Array.isArray(item[0]?.tool_call_chunks) || item[0].tool_call_chunks.length === 0)).toBe(true);
        expect(executed).toEqual([]);
        expect(requests).toHaveLength(2);
      }
    },
  );

  it.each(["length", "length-valid", "length-empty", "length-whitespace"] as const)(
    "does not execute or retry a %s response", async (scenario) => {
      for (const mode of ["invoke", "stream"] as const) {
        const { agent, requests, executed } = harness(scenario);
        const input = { messages: [{ role: "user" as const, content: "Record 123." }] };
        if (mode === "invoke") {
          await expect(agent.invoke(input)).rejects.toThrow("length stop");
        } else {
          await expect(async () => {
            const events = await agent.stream(input, { streamMode: "messages" });
            for await (const _event of events) { /* consume */ }
          }).rejects.toThrow("length stop");
        }
        expect(requests).toHaveLength(1);
        expect(executed).toEqual([]);
      }
    },
  );

  it("preserves the normal stream message shapes", async () => {
    const plain = harness("normal", false);
    const guarded = harness("normal", true, false, undefined, undefined, undefined, "fixture-key", undefined);
    const collect = async (agent: typeof plain.agent) => {
      const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
      const messages = [];
      for await (const [message] of stream) {
        messages.push({ type: message.type, content: message.content,
          ...(message.type === "ai" ? { tool_calls: (message as AIMessage).tool_calls } : {}) });
      }
      return messages;
    };
    expect(await collect(guarded.agent)).toEqual(await collect(plain.agent));
    expect(guarded.requests).toEqual(plain.requests);
  });

  it("streams original reasoning and keeps malformed arguments and correction reasoning private", async () => {
    const { agent, requests, executed } = harness("invalid");
    const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
    const output: unknown[] = [];
    for await (const item of stream) output.push(item);
    expect(requests).toHaveLength(3);
    expect(executed).toEqual([123]);
    expect(JSON.stringify(output)).not.toContain('{"amount":123,}');
    expect(JSON.stringify(output)).toContain("failure-only-reasoning");
    expect(JSON.stringify(output)).not.toContain("corrected-reasoning");
    const retry = requests[1]!.messages;
    expect(retry.filter((message: any) => message.role === "user")).toHaveLength(1);
    const assistant = retry.find((message: any) => message.role === "assistant");
    expect(assistant.reasoning_content).toBe("failure-only-reasoning");
  });

  it("recovers arguments split across streamed fragments", async () => {
    const { agent, requests, executed } = harness("split");
    const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
    for await (const _item of stream) { /* consume to completion */ }
    expect(requests).toHaveLength(3);
    const result = requests[1]!.messages.find((message: any) => message.role === "tool");
    expect(result.tool_call_id).toBe("original");
    expect(JSON.parse(result.content.split("\n")[1]).raw_arguments).toBe('{"amount":123,}');
    expect(executed).toEqual([123]);
  });

  it("delimits multiline arguments and reports the original character location", async () => {
    const { agent, requests, executed } = harness("multiline");
    const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
    for await (const _item of stream) { /* consume to completion */ }
    const feedback = requests[1]!.messages.find((message: any) => message.role === "tool");
    const lines = String(feedback.content).split("\n");
    expect(lines).toHaveLength(3);
    expect(JSON.parse(lines[1]!)).toEqual({
      raw_arguments: multilineArguments,
      parse_error: expect.stringMatching(/\(line 2, column 28\)$/),
    });
    expect(executed).toEqual([123]);
  });

  it("does not retry when the streamed tool ID is ambiguous", async () => {
    const { agent, requests, executed } = harness("repeated-id");
    await expect(async () => {
      const stream = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] }, { streamMode: "messages" });
      for await (const _item of stream) { /* consume */ }
    }).rejects.toThrow("ambiguous");
    expect(requests).toHaveLength(1);
    expect(executed).toEqual([]);
  });

  it("reports the failed model attempt to error callbacks without displaying arguments in the error message", async () => {
    const { agent } = harness("invalid");
    const errors: Error[] = [];
    await agent.invoke({ messages: [{ role: "user", content: "Record 123." }] }, {
      callbacks: [{ handleLLMError(error: Error) { errors.push(error); } }],
    });
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toBe("EXAONE returned malformed tool arguments");
    expect(Object.keys(errors[0]!)).not.toContain("details");
    expect(String(errors[0])).not.toContain('{"amount":123,}');
  });

  it("does not send a correction request after cancellation", async () => {
    const { agent, requests, executed } = harness("invalid");
    const controller = new AbortController();
    await expect(agent.invoke({ messages: [{ role: "user", content: "Record 123." }] }, {
      signal: controller.signal,
      callbacks: [{ handleLLMError() { controller.abort(); } }],
    })).rejects.toThrow();
    expect(requests).toHaveLength(1);
    expect(executed).toEqual([]);
  });

  it("preserves streamed text and never runs a tool after cancellation during correction", async () => {
    const controller = new AbortController();
    const invoked: number[] = [];
    let requests = 0;
    const record = tool(({ amount }) => { invoked.push(amount); return "done"; }, {
      name: "record", description: "Synthetic", schema: z.object({ amount: z.number() }),
    });
    const fetcher: typeof fetch = async (_url, init) => {
      requests++;
      if (requests === 2) {
        controller.abort();
        init?.signal?.throwIfAborted();
      }
      const frames = [
        { choices: [{ index: 0, delta: { content: "Original preamble." } }] },
        { choices: [{ index: 0, delta: { tool_calls: [{ ...call("original", '{"amount":123,}'), index: 0 }] },
          finish_reason: "tool_calls" }] },
      ];
      return new Response(frames.map((frame) => 'data: ' + JSON.stringify(frame) + '\n\n').join("") + "data: [DONE]\n\n");
    };
    const agent = createAgent({ model: new ChatExaone({ apiKey: "fixture", fetch: fetcher, maxRetries: 0 }),
      tools: [record], middleware: [createMiddleware(toolCallRecovery())] });
    const visible: unknown[] = [];
    await expect(async () => {
      const events = await agent.stream({ messages: [{ role: "user", content: "Record 123." }] },
        { signal: controller.signal, streamMode: "messages" });
      for await (const [chunk] of events) if (chunk.content) visible.push(chunk.content);
    }).rejects.toThrow();
    expect(visible).toEqual(["Original preamble."]);
    expect(requests).toBe(2);
    expect(invoked).toEqual([]);
  });

  it("isolates simultaneous guarded and unguarded invocations of the same model", async () => {
    const counts = new Map<string, number>();
    const choices = new Map<string, unknown[]>();
    const fetcher: typeof fetch = async (_url, init) => {
      const request = JSON.parse(String(init?.body));
      const prompt = request.messages.find((message: any) => message.role === "user").content as string;
      const count = (counts.get(prompt) ?? 0) + 1;
      counts.set(prompt, count);
      choices.set(prompt, [...(choices.get(prompt) ?? []), request.tool_choice]);
      await new Promise((resolve) => setTimeout(resolve, 5));
      const argumentsText = count === 1 ? '{"amount":123,}' : '{"amount":123}';
      return Response.json({ id: "fixture", model: "fixture", choices: [{ index: 0, finish_reason: "tool_calls",
        message: { role: "assistant", content: null, tool_calls: [call("same", argumentsText)] } }] });
    };
    const model = new ChatExaone({ apiKey: "fixture", fetch: fetcher, maxRetries: 0 });
    const guardedMessages = [new HumanMessage("guarded")];
    const plainMessages = [new HumanMessage("plain")];
    const record = tool(() => "Recorded.", { name: "record", description: "Synthetic", schema: z.object({ amount: z.number() }) });
    const guarded = toolCallRecovery().wrapModelCall(
      { model, messages: guardedMessages, tools: [record], toolChoice: "auto" as TestToolChoice },
      async (request) => await model.invoke(request.messages, { tools: request.tools, tool_choice: request.toolChoice }) as unknown as AIMessage);
    const [corrected, plain] = await Promise.all([guarded, model.invoke(plainMessages)]);
    expect(corrected.tool_calls).toHaveLength(1);
    expect(plain.invalid_tool_calls).toHaveLength(1);
    expect(counts.get("guarded")).toBe(2);
    expect(counts.get("plain")).toBe(1);
    expect(choices.get("guarded")).toEqual(["auto", namedRecord]);
    expect(choices.get("plain")).toEqual([undefined]);
  });

  it("recognizes a bindTools wrapper when called through the public middleware handler", async () => {
    const requests: Array<Record<string, unknown>> = [];
    const fetcher: typeof fetch = async (_url, init) => {
      requests.push(JSON.parse(String(init?.body)));
      return Response.json({ id: "fixture", model: "fixture", choices: [{ index: 0, finish_reason: "tool_calls",
        message: { role: "assistant", content: null, tool_calls: [call("original", requests.length === 1 ? '{"amount":123,}' : '{"amount":123}')] } }] });
    };
    const model = new ChatExaone({ apiKey: "fixture", fetch: fetcher, maxRetries: 0 });
    const record = tool(() => "Recorded.", { name: "record", description: "Synthetic", schema: z.object({ amount: z.number() }) });
    const bound = model.bindTools([record]);
    const messages = [new HumanMessage("synthetic prompt")];
    const result = await toolCallRecovery().wrapModelCall(
      { model: bound, messages, tools: [record], toolChoice: "auto" as TestToolChoice },
      async (request) => await bound.invoke(request.messages, { tool_choice: request.toolChoice }) as unknown as AIMessage);
    expect(requests.map((request) => request.tool_choice)).toEqual(["auto", namedRecord]);
    expect(result.tool_calls?.[0]?.args).toEqual({ amount: 123 });
  });

  it("does not feed a credential present in malformed arguments back to the model", async () => {
    let requests = 0;
    const fetcher: typeof fetch = async () => {
      requests++;
      return Response.json({ id: "fixture", model: "fixture", choices: [{ index: 0, finish_reason: "tool_calls",
        message: { role: "assistant", content: null, tool_calls: [call("original", '{"amount":"fixture%20key",}')] } }] });
    };
    const model = new ChatExaone({ apiKey: "fixture key", fetch: fetcher, maxRetries: 0 });
    const messages = [new HumanMessage("synthetic prompt")];
    await expect(toolCallRecovery().wrapModelCall({ model, messages }, async (request) =>
      await model.invoke(request.messages) as unknown as AIMessage))
      .rejects.toThrow("credentials");
    expect(requests).toBe(1);
  });
});
