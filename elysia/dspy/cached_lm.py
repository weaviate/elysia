import litellm
from dspy.clients.lm import LM, cached_litellm_completion, litellm_completion, cached_litellm_text_completion, litellm_text_completion
from typing import Optional, List, Any

import uuid
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional

class CachingLM(LM):
    def __init__(
        self,
        model: str,
        model_type: str = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        callbacks: Optional[List[Any]] = None,
        num_retries: int = 3,
        provider=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            **kwargs,
        )

    def __call__(self, prompt=None, messages=None, **kwargs):

        if messages is not None:
            for message in messages:
                if message["role"] == "system":
                    message["content"] = [
                        {
                            "type": "text",
                            "text": message["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]

        return super().__call__(prompt=prompt, messages=messages, **kwargs)


class CachingStreamingLM(LM):
    def __init__(
        self,
        model: str,
        model_type: str = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        callbacks: Optional[List[Any]] = None,
        num_retries: int = 3,
        provider=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            **kwargs,
        )

    async def __call__(self, prompt=None, messages=None, **kwargs):

        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        for message in messages:
            if message["role"] == "system":
                message["content"] = [
                    {
                        "type": "text",
                        "text": message["content"],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]

        # Make the request and handle LRU & disk caching.
        if self.model_type == "chat":
            completion = cached_litellm_completion if cache else litellm_completion
        else:
            completion = cached_litellm_text_completion if cache else litellm_text_completion

        response = completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            stream=True
        )

        chunks = []
        for chunk in response:
            chunks.append(chunk.choices[0].delta.content)
            yield chunk.choices[0].delta.content

        all_chunks = litellm.stream_chunk_builder(chunks, messages=messages)
        outputs = [c.message.content if hasattr(c, "message") else c["text"] for c in all_chunks["choices"]]

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(prompt=prompt, messages=messages, kwargs=kwargs, response=response)
        entry = dict(**entry, outputs=outputs, usage=dict(response["usage"]))
        entry = dict(**entry, cost=response.get("_hidden_params", {}).get("response_cost"))
        entry = dict(
            **entry,
            timestamp=datetime.now().isoformat(),
            uuid=str(uuid.uuid4()),
            model=self.model,
            model_type=self.model_type,
        )
        self.history.append(entry)
        self.update_global_history(entry)
            
        yield outputs
    

async def test():
    lm = CachingStreamingLM(model="gpt-4o-mini")
    async for chunk in lm(prompt="Hello, how are you?"):
        print(chunk)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())