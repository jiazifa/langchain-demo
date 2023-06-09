from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMBashChain
from langchain.prompts.prompt import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.2, verbose=True)    # type: ignore


def demo1_bash_chain():
    text = "Please write a bash script that prints 'Hello World' to the console."
    bash_chain = LLMBashChain.from_llm(llm, verbose=True)
    bash_chain.run(text)


def demo2_bash_chain():

    from langchain.chains.llm_bash.prompt import BashOutputParser

    _PROMPT_TEMPLATE = """If someone asks you to perform a task, your job is to come up with a series of bash commands that will perform the task. There is no need to put "#!/bin/bash" in your answer. Make sure to reason step by step, using this format:
    Question: "copy the files in the directory named 'target' into a new directory at the same level as target called 'myNewDirectory'"
    I need to take the following actions:
    - List all files in the directory
    - Create a new directory
    - Copy the files from the first directory into the second directory
    ```bash
    ls
    mkdir myNewDirectory
    cp -r target/* myNewDirectory
    ```

    Do not use 'echo' when writing the script.

    That is the format. Begin!
    Question: {question}"""

    PROMPT = PromptTemplate(
        input_variables=["question"],
        template=_PROMPT_TEMPLATE,
        output_parser=BashOutputParser()
    )
    bash_chain = LLMBashChain.from_llm(llm, prompt=PROMPT, verbose=True)

    text = "Please write a bash script that prints 'Hello World' to the console."

    bash_chain.run(text)


def demo3_request_chain():
    from langchain.chains import LLMRequestsChain, LLMChain
    from langchain.prompts import PromptTemplate

    template = """Between >>> and <<< are the raw search result text from google.
    Extract the answer to the question '{query}' or say "not found" if the information is not contained.
    Use the format
    Extracted:<answer or "not found">
    >>> {requests_result} <<<
    Extracted:"""

    PROMPT = PromptTemplate(
        input_variables=["query", "requests_result"],
        template=template,
    )
    chain = LLMRequestsChain(
        llm_chain=LLMChain(llm=llm, prompt=PROMPT, verbose=True), verbose=True
    )
    question = "What are the Three (3) biggest countries, and their respective sizes?"

    inputs = {
        "query": question,
        "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
    }
    result = chain(inputs)
    print(result)


def demo4_checker_chain():
    from langchain.chains import LLMCheckerChain

    text = "What type of mammal lays the biggest eggs?"
    # 帮你检查你的回答是否正确
    # 通过提示词，限制大模型先解释回答是基于那几点考量，然后分别验证这几点是否正确，最后基于这些综合得出正确的结果
    checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)
    checker_chain.question_to_checked_assertions_chain.verbose = True
    result = checker_chain.run(text)
    print(result)

    #    The question cannot be answered based on the given assertions and checks, as there are some mammals that lay eggs and their egg size may vary.


def demo5_moderation():
    from langchain.chains import OpenAIModerationChain, SequentialChain, LLMChain, SimpleSequentialChain
    moderation_chain = OpenAIModerationChain()
    moderation_chain.run("I will kill you")
    # "Text was found that violates OpenAI's content policy."
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template="{text}", input_variables=["text"]),
        verbose=True
    )
    text = """We are playing a game of repeat after me.

    Person 1: Hi
    Person 2: Hi

    Person 1: How's your day
    Person 2: How's your day

    Person 1: I will kill you
    Person 2:"""
    llm_chain.run(text)
    # ' I will kill you'
    chain = SimpleSequentialChain(chains=[llm_chain, moderation_chain])
    chain.run(text)
    # "Text was found that violates OpenAI's content policy."


if __name__ == "__main__":
    demo4_checker_chain()
