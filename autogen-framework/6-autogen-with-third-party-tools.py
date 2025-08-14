from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from pydantic import BaseModel

from autogen_core.models import UserMessage
import time
import asyncio
from autogen_ext.tools.http import HttpTool
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(api_key=key)

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


model_client=OpenAIChatCompletionClient(model='gpt-4o',api_key=api_key)

## THIRD PARTY TOOL INTEGRATION
def web_search(param: str) -> str:
    """Search the web for the given query and return the results."""
    try:
        results = tavily_tool.run(param)
        return results
    except Exception as e:
        print(f"Error occurred while searching the web: {e}")
        return "No results found."


## AGENT INITIALIZATION
agent = AssistantAgent(name="fetcher", 
                       model_client=model_client, 
                       system_message="You are a helpful assistant that can search the web for information using the search_web tool." \
    "Please make sure that you use the search_web tool to find information before you return the answer." \
    "don't send the year in query, rather use latest or recently etc.",
                       tools=[web_search], 
                       reflect_on_tool_use=True # (WITH TRUE - The reversed string is "!dlroW ,olleH".   WITH FALSE - !dlroW ,olleH)
                       )

async def main():
    result = await agent.run(task='Who won the IPL recently ?')
    # print(result)
    print("==========================")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
