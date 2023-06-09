from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

with open("./examples/state_of_the_union.txt") as f:
    state_of_the_union = f.read()

llm = ChatOpenAI(temperature=0.2)    # type: ignore


def demo1():

    def transform_func(inputs: dict) -> dict:
        text = inputs["text"]
        shortened_text = "\n\n".join(text.split("\n\n")[:3])
        return {"output_text": shortened_text}

    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["output_text"],
        transform=transform_func
    )

    template = """Summarize this text:

    {output_text}

    Summary:"""
    prompt = PromptTemplate(input_variables=["output_text"], template=template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)    # type: ignore
    # 只总结了前三段
    sequential_chain = SimpleSequentialChain(
        chains=[transform_chain, llm_chain], verbose=True
    )

    result = sequential_chain.run(state_of_the_union)
    print(result)


def demo2():
    # 通过TransformChain来构造一个简单的摘要， 很费钱费时
    # map_reduce 是先分段总结，然后总结所有的总结
    # refine: 总结 + 新片段 => 新总结
    # stuff: 全集合起来，一起总结
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chains import AnalyzeDocumentChain

    summary_chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
    summarize_document_chain = AnalyzeDocumentChain(
        combine_docs_chain=summary_chain, verbose=True
    )
    result = summarize_document_chain.run(state_of_the_union)
    print(result)
    # President Biden addressed Congress and the nation on a range of issues, including Russian aggression towards Ukraine, economic relief for Americans, rebuilding infrastructure, fighting inflation, closing tax loopholes for the wealthy, combating COVID-19, reducing gun violence, protecting voting rights, and supporting veterans. He emphasized the importance of unity and bipartisanship in tackling these challenges and expressed optimism about America's future.


def demo3():
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import AnalyzeDocumentChain
    # 通过访问每一段，试图找出和问题相关的内容，然后在最后一轮问答中，将前面总结的内容附上，再问答
    qa_chain = load_qa_chain(llm, chain_type="map_reduce", verbose=True)
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain, verbose=True)
    qa_document_chain.run(
        input_document=state_of_the_union,
        question="what did the president say about justice breyer?"
    )


if __name__ == "__main__":
    demo3()
