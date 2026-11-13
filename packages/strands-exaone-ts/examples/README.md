# Strands examples

Each directory is a separate runnable example with its own setup instructions. Install only the framework you want to try. Additional framework dependencies are installed only for the corresponding examples.

| Example | Demonstrates |
| --- | --- |
| [Strands SDK](sdk/) | Tool execution, streamed conversation history, and structured output |
| [Strands Harness](harness/) | Custom tools, conversation history, opt-in file reading and search |
| [Strands CLI](strands-cli/) | Use the provider from the Strands CLI |

Set `EXAONE_API_KEY` in your environment. Run examples one at a time within your account limits. They use synthetic tasks, but requests still consume API quota. See each directory for tool permissions and filesystem behavior.
