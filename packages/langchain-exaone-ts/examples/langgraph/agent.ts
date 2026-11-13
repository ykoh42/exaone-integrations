import { SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MemorySaver, MessagesValue, START, StateGraph, StateSchema } from "@langchain/langgraph";
import type { GraphNode } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import { ChatExaone } from "langchain-exaone";

const multiply = tool(({ a, b }) => String(a * b), {
  name: "multiply",
  description: "Multiply two integers.",
  schema: z.object({ a: z.number().int(), b: z.number().int() }),
});
const model = new ChatExaone({ baseUrl: process.env.EXAONE_BASE_URL ?? "https://api.exaone.ai/v1" })
  .bindTools([multiply]);
const State = new StateSchema({ messages: MessagesValue });

const callModel: GraphNode<typeof State> = async (state) => ({
  messages: [await model.invoke([
    new SystemMessage("Use the multiply tool for multiplication."),
    ...state.messages,
  ])],
});

const agent = new StateGraph(State)
  .addNode("model", callModel)
  .addNode("tools", new ToolNode([multiply]))
  .addEdge(START, "model")
  .addConditionalEdges("model", toolsCondition)
  .addEdge("tools", "model")
  .compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: "calculator" }, recursionLimit: 20 };
for (const prompt of ["What is 6 times 7?", "Multiply that result by 2."]) {
  const result = await agent.invoke({ messages: [{ role: "user", content: prompt }] }, config);
  console.log(result.messages.at(-1)?.text);
}
