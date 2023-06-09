import os
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0),    # type: ignore
    verbose=True
)
path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(path, "serialization_llm_chain.json")
llm_chain.save(file_path)

# from langchain.chains import load_chain

    # chain = load_chain(file_path)
