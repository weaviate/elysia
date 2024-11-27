import json

# Objects
from elysia.api.objects import Update

class Text:
    def __init__(self, objects: list[dict], metadata: dict = {}):
        self.objects = objects
        self.metadata = metadata
        self.type = "text"

    def to_json(self):
        return {
            "type": self.type,
            "metadata": self.metadata,
            "objects": self.objects
        }

    def to_str(self):
        return json.dumps({
            "type": self.type,
            "metadata": self.metadata,
            "objects": self.objects
        })
    
    def to_llm_str(self):
        return self.to_str()
    
    def return_value(self):
        return self.text

    def to_frontend(self, conversation_id: str, query_id: str = None):
        return Update.to_frontend_json(
            "text",
            conversation_id,
            query_id,
            self.to_json()
        )

class Response(Text):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "response"

class Summary(Text):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "summary"