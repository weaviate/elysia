from typing import List
from pydantic import BaseModel, Field
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
    cited_text: List[TextWithCitation] = Field(
        description="A list of TextWithCitation objects"
    )

    @classmethod
    def is_streamable(cls):
        return True
