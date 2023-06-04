import typing
import warnings
from pydantic import root_validator, Field

from langchain.schema import BaseMessage, ChatResult, HumanMessage, AIMessage, ChatGeneration
from transformers import AutoModel, AutoTokenizer
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chat_models.base import BaseChatModel
from langchain.chains import ConversationChain


class ChatGLM(BaseChatModel):
    streaming: bool = False
    max_token: typing.Optional[int] = None
    temperature: float = 0.2
    model_name: str = Field(default="THUDM/chatglm-6b-int8", alias="model")
    client: typing.Any    #: :meta private:
    tokenizer: typing.Any    #: :meta private:

    @root_validator()
    def raise_deprecation(cls, values: typing.Dict) -> typing.Dict:
        # model path
        # default_model_name = "THUDM/chatglm-6b-int8"
        # model_path_or_name: typing.Optional[str] = values.get("model_path_or_name")
        # if not model_path_or_name:
        #     model_path_or_name = default_model_name
        #     warnings.warn(
        #         f"model_path_or_name using default value: {default_model_name}"
        #     )
        # trust_remote = values.get("trust_remote_code", True)
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_path_or_name, trust_remote_code=trust_remote
        # )
        # model = AutoModel.from_pretrained(
        #     model_path_or_name, trust_remote_code=trust_remote
        # ).half().cuda()
        # model.eval()
        # values["model"] = model
        # values["tokenizer"] = tokenizer
        return values

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _generate(
        self,
        messages: typing.List[BaseMessage],
        stop: typing.List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None
    ) -> ChatResult:
        params = self._create_message_dicts(messages)
        if self.streaming:
            inner_completion = ""
            for response, _ in self.client.stream_chat(tokenizer, **params):
                token = response
                inner_completion += token
                if run_manager:
                    run_manager.on_llm_new_token(token=token)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        text=inner_completion,
                        message=AIMessage(content=inner_completion)
                    )
                ]
            )
        response, _ = self.client.chat(self.tokenizer, **params)
        return ChatResult(
            generations=[
                ChatGeneration(text=response, message=AIMessage(content=response))
            ]
        )

    async def _agenerate(
        self,
        messages: typing.List[BaseMessage],
        stop: typing.List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None
    ) -> ChatResult:
        raise NotImplementedError

    def _create_message_dicts(self, messages: typing.List[BaseMessage]) -> typing.Dict:
        params: typing.Dict[str, typing.Any] = {}
        params["query"] = messages[-1].content
        if len(messages) > 1:
            histories: typing.List[typing.Tuple[str, str]] = []
            human_input: typing.Optional[str] = None
            for _, m in enumerate(messages[:-1]):
                if m.type == "human":
                    human_input = m.content
                    continue
                if m.type == "ai" and human_input:
                    histories.append((human_input, m.content))
                    human_input = None

            params["history"] = histories
        else:
            params["history"] = []
        params["temperature"] = self.temperature
        return params


if __name__ == "__main__":

    model_path = "THUDM/chatglm-6b-int8"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    client = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    client.eval()
    llm = ChatGLM(client=client, tokenizer=tokenizer)    #type: ignore
    messages: typing.List[BaseMessage] = []
    messages.append(HumanMessage(content="你好"))
    messages.append(AIMessage(content="你好！有什么我可以帮助你的吗？"))
    history = ChatMessageHistory(messages=messages)
    memory = ConversationBufferMemory(chat_memory=history)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    messages.append(HumanMessage(content="写一首关于春天的绝句"))
    result = llm._generate(messages=messages)
    print(f"result: {result.generations[0].text}")
    # output = conversation.predict(input="写一首关于春天的绝句")
    # print(f"output: {output}")
