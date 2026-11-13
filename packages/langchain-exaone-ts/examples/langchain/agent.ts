import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph";
import { createAgent } from "langchain";
import { z } from "zod";
import { ChatExaone } from "langchain-exaone";

const multiply = tool(({ a, b }) => String(a * b), {
  name: "multiply",
  description: "Multiply two integers.",
  schema: z.object({ a: z.number().int(), b: z.number().int() }),
});

const agent = createAgent({
  model: new ChatExaone({ baseUrl: process.env.EXAONE_BASE_URL ?? "https://api.exaone.ai/v1" }),
  tools: [multiply],
  systemPrompt: "Use the multiply tool for multiplication.",
  checkpointer: new MemorySaver(),
});

const config = { configurable: { thread_id: "calculator" }, recursionLimit: 20 };
for (const prompt of ["What is 6 times 7?", "What numbers did you multiply? Answer in Korean."]) {
  const result = await agent.invoke({ messages: [{ role: "user", content: prompt }] }, config);
  console.log(result.messages.at(-1)?.text);
}
