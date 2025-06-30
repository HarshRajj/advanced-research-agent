from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools 
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import asyncio
import os
load_dotenv()



from pydantic import SecretStr

model = ChatGroq(
        model = "deepseek-r1-distill-llama-70b", 
        temperature = 0.1,
        max_tokens = 512,
        api_key = SecretStr(os.getenv("GROQ_API_KEY")) 

)

server_params= StdioServerParameters(
    command = "npx",
    env = {
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
        
    args = ["firecrawl-mcp"]

)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(
                model=model,
                tools=tools,
                
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful agent that can scrape websites and answer questions based on the content you find."
                }
            ]
            print("available tools : ", *[tool.name for tool in tools])
            print("--"*50)

            while True:
                user_input = input("You:")
                if user_input.lower() in ["exit", "quit"]:
                    break

                messages.append({"role": "user", "content":user_input})

                try:
                    short_messages = messages[-1]
                    agent_response= await agent.ainvoke({"messages":short_messages})
                    ai_message = agent_response["messages"][-1].content
                    # messages.append({"role": "assistant", "content": ai_message})
                    print(f"AI: {ai_message}")
        
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())



