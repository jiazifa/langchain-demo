from abc import ABC
import typing
from pydantic import root_validator
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.llms.base import LLM

from transformers import AutoModel, AutoTokenizer


class GLM(LLM, ABC):

    max_token: int = 4096
    temperature: float = 0.2
    model: typing.Any = None    # type: ignore
    tokenizer: typing.Optional[AutoTokenizer] = None

    @root_validator()
    def validate_environment(cls, values: typing.Dict) -> typing.Dict:
        # model path
        model_path = "THUDM/chatglm-6b-int8"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path,
                                          trust_remote_code=True).half().cuda()
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
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    async def _acall(
        self,
        prompt: str,
        stop: typing.List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None
    ) -> str:

        return ""
