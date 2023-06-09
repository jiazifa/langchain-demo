from langchain.llms import HuggingFacePipeline

model_id = "TheBloke/vicuna-7B-1.1-HF"
llm = HuggingFacePipeline.from_model_id(
    model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_length": 2048,
        "temperature": 0,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
    }
)
result = llm._call("Hello")
print(result)
