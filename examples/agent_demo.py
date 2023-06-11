import os
import typing
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
    from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_vectorstore_agent
    from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
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

    def on_csv_agent_task(path: str, query: str) -> str:
        agent: AgentExecutor = create_csv_agent(llm=llm, path=path)
        return agent.run(query)

    csv_tool1 = Tool(
        name="Billing Information",
        func=lambda x: on_csv_agent_task("data/billing.csv", x),
        description=
        "useful when you want to know about personal billing, like 'How much is the total expenditure?' or 'How much did the catering cost?'",
    )

    def on_self_infomation_tool(query: str) -> str:
        documents: typing.List[Document] = []
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                with open(os.path.join("data", file), "r") as f:
                    documents.append(
                        Document(page_content=f.read(), metadata={"name": file})
                    )
        print(f"documents: {documents}")
        db = Chroma.from_documents(documents, embedding=default_embedding)
        vectorstore_info = VectorStoreInfo(
            name="state_of_union_address",
            description="the most recent state of the Union adress",
            vectorstore=db
        )
        return db.similarity_search(query, k=1)[0].page_content

    self_infomation_tool = Tool(
        name="Self Information Query",
        func=lambda x: on_self_infomation_tool(x),
        description=
        "useful when you want to know about personal information, like 'What is my name?' or 'What is my address?'"
    )

    ALL_TOOLS = [search_tool] + [csv_tool1] + [self_infomation_tool]
    documents = [
        Document(page_content=t.description, metadata={
            "name": t.name,
            "index": i
        }) for i, t in enumerate(ALL_TOOLS)
    ]
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    default_embedding = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    store = Chroma.from_documents(documents, embedding=default_embedding)
    retriever = store.as_retriever()
    docs = retriever.get_relevant_documents("whats my name")
    docs = docs[:1]
    d = [ALL_TOOLS[d.metadata["index"]] for d in docs]
    initialize_agent(
        ALL_TOOLS, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    ).run("who am i?")


if __name__ == "__main__":
    demo_retriever_tool()
