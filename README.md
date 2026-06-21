# Venture Dive Customer Agent

A customer support chat agent for Venture Dive, built with LangChain and Streamlit.

## Features

- **Order details**: look up order status by order ID
- **Return orders**: initiate returns for delivered orders
- **Multi-turn context**: the agent reasons across the full conversation history
- **Persona**: A Venture Dive support agent

## Run

```bash
cd src
uv sync
uv run streamlit run app.py
```

Requires an `OPENAI_API_KEY` in `src/.env`.

## Screenshots

![Demo](demo/demo.jpg)

![Demo 2](demo/demo2.jpg)

![Demo 3](demo/demo3.jpg)
