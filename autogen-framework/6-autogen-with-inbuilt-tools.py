from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from pydantic import BaseModel

from autogen_core.models import UserMessage
import time
import asyncio
from autogen_ext.tools.http import HttpTool


load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


model_client=OpenAIChatCompletionClient(model='gpt-4o',api_key=api_key)

## INBUILT TOOL CALLED
# Define a JSON schema for a base64 decode tool
schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "A random cat fact"
            },
            "length": {
                "type": "integer",
                "description": "Length of the cat fact"
            }
        },
        "required": ["fact", "length"],
    }

# Create an HTTP tool for the httpbin API
http_tool = HttpTool(
    name="cat_facts_api",
    description="get a cool cat fact",
    scheme="https",
    host="catfact.ninja",
    port=443,
    path="/fact",
    method="GET",
    return_type="json",
    json_schema= schema
)


## AGENT INITIALIZATION
agent = AssistantAgent(name="fetcher", 
                       model_client=model_client, 
                       system_message='You are a helpful assistant that can provide cat facts using the cat_facts_api tool. Give the result with summary',
                       tools=[http_tool], 
                       reflect_on_tool_use=True # (WITH TRUE - The reversed string is "!dlroW ,olleH".   WITH FALSE - !dlroW ,olleH)
                       )

async def main():
    result = await agent.run(task='Get me a fact about cat')
    print(result)
    print("==========================")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
