import os
import typing
import logging
from abc import ABC
from pydantic import BaseModel, Field, PrivateAttr, root_validator
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.docstore.base import Document

logger = logging.getLogger(__name__)


class VDatabase(BaseModel, ABC):

    persist_directory: str = Field(..., env="VECTORSTORE_ROOT")

    collection_name: str = Field(default="default")

    embeddings: Embeddings = Field(default=None)

    _db: Chroma = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._db = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    @root_validator(pre=True)
    def prepare_datas(cls, values: typing.Dict) -> typing.Dict:
        if "persist_directory" not in values:
            logger.warning("persist_directory is not set, reading from env")
            if "VECTORSTORE_ROOT" not in os.environ:
                raise ValueError("VECTORSTORE_ROOT is not set")
            values["persist_directory"] = os.environ["VECTORSTORE_ROOT"]

        if "collection_name" not in values:
            logger.warning("collection_name is not set, use default")
            values["collection_name"] = "my_collection"

        if "embeddings" not in values:
            logger.warning("embeddings is not set, use default")
            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            default_embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            values["embeddings"] = default_embedding

        return values

    def insert(self, document: typing.List[Document]) -> typing.List[str]:
        return self._db.add_documents(document)

    async def ainsert(self, document: typing.List[Document]) -> typing.List[str]:
        return await self._db.aadd_documents(document)

    def update(self, document_id: str, document: Document) -> None:
        return self._db.update_document(document_id=document_id, document=document)

    def query(
        self, query: str, search_type: typing.Literal["similarity", 'mmr'], **kwargs
    ) -> typing.List[Document]:
        return self._db.search(query, search_type, **kwargs)

    async def aquery(self, query: str, search_type: str,
                     **kwargs) -> typing.List[Document]:
        return await self._db.asearch(query, search_type, **kwargs)

    def persist(self) -> None:
        self._db.persist()
