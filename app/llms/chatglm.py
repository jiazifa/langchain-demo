from abc import ABC
import typing
from pydantic import root_validator
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoModel, AutoTokenizer


class ChatGLM(LLM, ABC):

    max_token: int = 4096
    temperature: float = 0.2
    model_path_or_name: str = "THUDM/chatglm-6b"
    model: typing.Any    # type: ignore
    tokenizer: AutoTokenizer

    @root_validator()
    def raise_deprecation(cls, values: typing.Dict) -> typing.Dict:
        # model path
        if model_path := values.get("model_path_or_name"):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).half().cuda()
            model.eval()
            values["model"] = model
            values["tokenizer"] = tokenizer
        return values

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _call(
        self,
        prompt: str,
        stop: typing.List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None
    ) -> str:
        return ""

    async def _acall(
        self,
        prompt: str,
        stop: typing.List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None
    ) -> str:

        return ""