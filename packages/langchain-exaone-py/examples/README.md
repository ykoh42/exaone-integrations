# LangChain examples

Each directory has its own setup instructions and installs only the dependencies for that framework.

| Example | Demonstrates |
| --- | --- |
| [LangChain](langchain/) | Streaming chat, tool execution, checkpoints, and structured output |
| [LangGraph](langgraph/) | Explicit tool loop and conversation checkpoints |
| [Deep Agents](deepagents/) | Planning, EXAONE subagents, and a virtual-file report |

Set `EXAONE_API_KEY` in your environment and run examples one at a time within your account limits. The synthetic tasks make real API requests and consume quota. Each directory explains its tool permissions and filesystem behavior.
