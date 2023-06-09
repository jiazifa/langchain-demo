from abc import ABC
import typing
from pydantic import Field, PrivateAttr
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms import HuggingFacePipeline
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, Pipeline

from transformers import AutoModel, AutoTokenizer


class Llama(LLM, ABC):
    model_id: str = Field(default="TheBloke/vicuna-7B-1.1-HF")

    tokenizer: LlamaTokenizer = PrivateAttr()

    model: LlamaForCausalLM = PrivateAttr()

    pipeline: Pipeline = PrivateAttr()

    llm: HuggingFacePipeline = PrivateAttr()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tokenizer = LlamaTokenizer.from_pretrained(self.model_id)

        model = LlamaForCausalLM.from_pretrained(
            self.model_id,
        #   load_in_8bit=True, # set these options if your GPU supports them!
        #   device_map=1#'auto',
        #   torch_dtype=torch.float16,
        #   low_cpu_mem_usage=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        self.llm = local_llm
