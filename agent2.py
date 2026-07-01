import json
import os
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END, START
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # type: ignore

MEMORY_FILE = "memory.json"


class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]


load_dotenv()

llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    provider="together",
)

model = ChatHuggingFace(llm=llm)


def load_memory(state: AgentState) -> AgentState:
    if not os.path.exists(MEMORY_FILE):
        return state
    with open(MEMORY_FILE, "r") as f:
        content = f.read().strip()
    if not content:
        return state
    data = json.loads(content)
    history = []
    for entry in data:
        if entry["role"] == "human":
            history.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "ai":
            history.append(AIMessage(content=entry["content"]))
    state["message"] = history + state["message"]
    return state


def process(state: AgentState) -> AgentState:
    res = model.invoke(state["message"])
    state["message"].append(AIMessage(content=res.content))
    print(f"\nAI message: {res.content}")
    return state


def save_memory(state: AgentState) -> AgentState:
    data = []
    for msg in state["message"]:
        if isinstance(msg, HumanMessage):
            data.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data.append({"role": "ai", "content": msg.content})
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return state


graph = StateGraph(AgentState)
graph.add_node("load_memory", load_memory)
graph.add_node("processor", process)
graph.add_node("save_memory", save_memory)
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "processor")
graph.add_edge("processor", "save_memory")
graph.add_edge("save_memory", END)

agent = graph.compile()
convo_history = []

user_input = input("User Input: ")
while user_input != "exit":
    convo_history.append(HumanMessage(content=user_input))
    res = agent.invoke({"message": convo_history})
    convo_history = res["message"]
    user_input = input("User Input: ")
