from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from pydantic import BaseModel

from autogen_core.models import UserMessage
import time
import asyncio

from autogen_core.tools import FunctionTool

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


model_client=OpenAIChatCompletionClient(model='gpt-4o',api_key=api_key)

## CUSTOM TOOL CREATED
def reverse_string(text: str) -> str:
    '''
    Reverse the given text

    input:str

    output:str

    The reverse string is returned.
    '''
    return text[::-1]


## BIND CUSTOM TOOL TO THE BASE TOOL LIST
reverse_tool = FunctionTool(reverse_string,description='A tool to reverse a string')

## AGENT INITIALIZATION
agent = AssistantAgent(name="fetcher", 
                       model_client=model_client, 
                       system_message='You are a helpful assistant that can reverse string using reverse_string tool. Give the result with summary', 
                       tools=[reverse_tool], 
                       reflect_on_tool_use=True # (WITH TRUE - The reversed string is "!dlroW ,olleH".   WITH FALSE - !dlroW ,olleH)
                       )

async def main():
    result = await agent.run(task='Reverse the string "Hello, World!"')
    print(result)
    print("==========================")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
