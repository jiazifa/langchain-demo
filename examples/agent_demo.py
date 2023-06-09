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

    agent: AgentExecutor = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

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
            func=lambda x:
            "'All I Want For Christmas Is You' by Mariah Carey.",    #Mock Function
            description=
            "A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
        )
    ]
    agent = initialize_agent(
        tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
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


def demo_retriever_tool():
    from langchain.agents import create_csv_agent
    from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
    from langchain.prompts import StringPromptTemplate
    from langchain import OpenAI, SerpAPIWrapper, LLMChain
    from typing import List, Union
    from langchain.schema import AgentAction, AgentFinish, Document
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    import re

    search_tool = Tool(
        name="Search",
        func=lambda x: f"im searching for {x}",
        description="useful for when you need to answer questions about current events"
    )

    csv_tool1 = Tool(
        name="csv-person-information",
        func=lambda x: "im ani, im 20 years old, im a student",
        description=
        "useful when you want to know about a person, like 'what is your name?' or 'how old are you?'"
    )

    ALL_TOOLS = [search_tool] + [csv_tool1]
    documents = [
        Document(page_content=t.description, metadata={"index": t.name})
        for _, t in enumerate(ALL_TOOLS)
    ]
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    default_embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    store = Chroma.from_documents(documents, embedding=default_embedding)
    retriever = store.as_retriever()
    docs = retriever.get_relevant_documents("whats your name")
    d = []


if __name__ == "__main__":
    demo_retriever_tool()
