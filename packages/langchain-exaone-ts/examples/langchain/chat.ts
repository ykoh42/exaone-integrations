import { AIMessageChunk, HumanMessage } from "@langchain/core/messages";
import { ChatExaone } from "langchain-exaone";

const model = new ChatExaone({ baseUrl: process.env.EXAONE_BASE_URL ?? "https://api.exaone.ai/v1" });
const question = new HumanMessage("Explain context windows in two sentences.");
let response = new AIMessageChunk({ content: "" });

for await (const chunk of await model.stream([question])) {
  process.stdout.write(chunk.text);
  response = response.concat(chunk);
}
console.log();

const followUp = await model.invoke([
  question,
  response,
  new HumanMessage("Summarize your answer in Korean."),
]);
console.log(followUp.text);
