from typing import List
from pydantic import BaseModel, Field, model_serializer
from dspy import Type


class TextWithCitation(BaseModel):
    text: str = Field(description="The text within the summary")
    ref_ids: List[str] = Field(
        description=(
            "The ref_ids of the citations relevant to the text. "
            "Can be an empty list if the text is not related to any of the citations."
        ),
        default_factory=list,
    )


class ListTextWithCitation(Type):
    objects: List[TextWithCitation] = Field(
        description="A list of TextWithCitation objects, containing the text and the ref_ids."
    )

    @classmethod
    def is_streamable(cls):
        return True

    @model_serializer()
    def serialize_model(self):
        return [t.model_dump() for t in self.objects]
