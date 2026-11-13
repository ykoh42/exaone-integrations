import { z } from "zod";
import { ChatExaone } from "langchain-exaone";

const model = new ChatExaone({ baseUrl: process.env.EXAONE_BASE_URL ?? "https://api.exaone.ai/v1" });
const Contact = z.object({
  name: z.string().describe("The person's name"),
  email: z.string().describe("The person's email address"),
});

const contact = await model.withStructuredOutput(Contact, { name: "Contact" }).invoke(
  "Extract the contact: Alex, alex@example.com."
);
console.log(contact);
