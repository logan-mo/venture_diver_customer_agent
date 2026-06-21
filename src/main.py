from tools import search_catalog, get_order_details, initiate_return
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

    agent = create_agent(llm, [search_catalog, get_order_details, initiate_return])

    input_message = {
        "role": "user",
        "content": "I want to return my laptop that I ordered last week. Can you help me with that?",
    }

    stream = agent.stream_events({"messages": [input_message]}, version="v3")

    for event in stream:
        if event["type"] == "tool_call":
            print(f"Tool called: {event['tool_name']} with args: {event['tool_args']}")
        elif event["type"] == "message":
            print(f"Agent message: {event['message']['content']}")
        elif event["type"] == "agent_finish":
            print("Agent finished execution.")
        else:
            print(f"Other event: {event}")


if __name__ == "__main__":
    main()
