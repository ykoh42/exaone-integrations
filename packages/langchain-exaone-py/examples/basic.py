"""Call EXAONE with the LangChain chat model integration."""

from langchain_exaone import ChatEXAONE

model = ChatEXAONE(reasoning_effort="low")
response = model.invoke("K-EXAONE을 한 문장으로 소개해 주세요.")

print(response.text)
print("Finish reason:", response.response_metadata.get("finish_reason"))
print("Usage:", response.usage_metadata)
