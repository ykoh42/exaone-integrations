import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph";
import { createDeepAgent } from "deepagents";
import { z } from "zod";
import { ChatExaone } from "langchain-exaone";

const getReleaseNotes = tool(
  () => "Release 1.0: Added Korean documentation and a typed calculator tool.",
  {
    name: "get_release_notes",
    description: "Read the synthetic release notes for this example.",
    schema: z.object({}),
  }
);

const model = new ChatExaone({ baseUrl: process.env.EXAONE_BASE_URL ?? "https://api.exaone.ai/v1" });
const agent = await createDeepAgent({
  model,
  systemPrompt: "Use the supplied release notes as your only factual source. Write concise reports.",
  tools: [getReleaseNotes],
  subagents: [{
    name: "release-reviewer",
    description: "Read and summarize the example release notes.",
    systemPrompt: "Call get_release_notes and summarize the changes accurately.",
    model,
    tools: [getReleaseNotes],
  }],
  checkpointer: new MemorySaver(),
});

const result = await agent.invoke(
  { messages: [{ role: "user", content:
    "Plan a release brief. Delegate reading the release notes to release-reviewer. " +
    "Write a short Korean brief to /release-brief.md, then summarize what you saved.",
  }] },
  { configurable: { thread_id: "release-brief" }, recursionLimit: 40 }
);
console.log(result.messages.at(-1)?.text);
console.log("Virtual files:", Object.keys(result.files ?? {}).sort());
