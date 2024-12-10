from dspy.clients.lm import LM
from typing import Optional, List, Any

class CachingLM(LM):
    def __init__(
        self,
        model: str,
        cache_key: Optional[str] = None,
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