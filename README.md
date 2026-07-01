# AI Agent

A conversational AI agent built with **LangGraph** and **LangChain**, powered by **Meta Llama 3.3 70B** via the Together AI provider.

## Agents

### agent1.py — Basic Agent
A simple single-node LangGraph agent. Takes user input in a loop, sends it to the LLM, and prints the response. No conversation history — each message is independent.

### agent2.py — Agent with Persistent Memory
An improved agent with a 3-node graph that remembers conversations across sessions.

```
START → load_memory → processor → save_memory → END
```

- **load_memory** — reads previous conversation from `memory.json` on startup
- **processor** — sends the full conversation history to the LLM and gets a response
- **save_memory** — writes the updated conversation back to `memory.json`

Conversation history persists between runs. Type `exit` to quit.

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/hydraveer/AI_Agent.git
cd AI_Agent
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirenment.txt
```

**4. Add your API key**

Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

**Run the basic agent:**
```bash
python agent1.py
```

**Run the agent with memory:**
```bash
python agent2.py
```

Type `exit` to end the session. For `agent2`, the conversation is saved to `memory.json` and reloaded automatically on the next run.

## Stack

| Library | Purpose |
|---|---|
| LangGraph | Agent graph orchestration |
| LangChain | LLM interface & message types |
| LangChain HuggingFace | HuggingFace model integration |
| Llama 3.3 70B | Underlying language model |
| Together AI | Inference provider |
| python-dotenv | API key management |
