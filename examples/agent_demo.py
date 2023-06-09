from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType

from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.2, verbose=True)    # type: ignore

def demo_base_tool():
    tools = load_tools(["llm-math"], llm=llm)

    agent: AgentExecutor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run("waht is the result of 1 + 1 ?")
    # > Entering new AgentExecutor chain...
    # This is a simple addition problem.
    # Action: Calculator
    # Action Input: 1 + 1
    # Observation: Answer: 2
    # Thought:That was easy.
    # Final Answer: 2

    # > Finished chain.

def demo_custom_tool():
    tools = [
        Tool(
            name="Music Search",
            func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.", #Mock Function
            description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
        )
    ]
    agent = initialize_agent(tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("what is the most famous song of christmas")
    # > Entering new AgentExecutor chain...
    # I should use Music Search to find the answer
    # Action: Music Search
    # Action Input: "most famous christmas song"
    # Observation: 'All I Want For Christmas Is You' by Mariah Carey.
    # Thought:I should double check this answer with another source
    # Action: Google Search
    # Action Input: "most famous christmas song"
    # Observation: Google Search is not a valid tool, try another one.
    # Thought:I should use Music Search again to confirm the answer
    # Action: Music Search
    # Action Input: "most famous christmas song"
    # Observation: 'All I Want For Christmas Is You' by Mariah Carey.
    # Thought:I now know the final answer
    # Final Answer: The most famous Christmas song is 'All I Want For Christmas Is You' by Mariah Carey.

    # > Finished chain.


if __name__ == "__main__":
    demo_custom_tool()
