# Langchain examples

Each directory is a separate runnable example with its own setup instructions. Install only the framework you want to try. Additional framework dependencies are installed only for the corresponding examples.

| Example | Demonstrates |
| --- | --- |
| [LangChain](langchain/) | Streaming chat, tool execution, checkpoints, and structured output |
| [LangGraph](langgraph/) | Explicit tool loop and conversation checkpoints |
| [Deep Agents](deepagents/) | Planning, EXAONE subagents, and a virtual-file report |

Set `EXAONE_API_KEY` in your environment. Run examples one at a time within your account limits. They use synthetic tasks, but requests still consume API quota. See each directory for tool permissions and filesystem behavior.
