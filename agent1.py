from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # type: ignore
from IPython.display import display, Image #type: ignore


load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    provider="together",
)

llm = ChatHuggingFace(llm=llm)


def process(state: AgentState) -> AgentState:
    res = llm.invoke(state["messages"])
    print(f"\nAI : {res.content if hasattr(res, 'content') else res}")  # type: ignore[union-attr]
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()
display(Image(agent.get_graph().draw_mermaid_png()))

user_input = input("Enter something: ")
while user_input!= "exit":
    
    messages = HumanMessage(content=user_input)

    agent.invoke({
        "messages": [messages]
    })
    user_input = input("Enter something: ")
