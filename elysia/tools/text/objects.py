from elysia.objects import Text
from typing import List
from pydantic import BaseModel, Field


class TextWithCitation(BaseModel):
    text: str = Field(description="The text within the summary")
    ref_ids: List[str] = Field(
        description=(
            "The ref_ids of the citations relevant to the text. "
            "Can be an empty list if the text is not related to any of the citations."
        )
    )


class TextWithTitle(Text):
    def __init__(self, text: str, title: str, **kwargs):
        Text.__init__(
            self,
            "text_with_title",
            [{"text": text}],
            metadata={"title": title},
            **kwargs,
        )


class TextWithCitations(Text):
    def __init__(self, cited_texts: List[TextWithCitation], title: str, **kwargs):
        Text.__init__(
            self,
            "text_with_citations",
            [
                {
                    "text": cited_text_item.text,
                    "ref_ids": cited_text_item.ref_ids,
                }
                for cited_text_item in cited_texts
            ],
            metadata={"title": title},
            **kwargs,
        )
