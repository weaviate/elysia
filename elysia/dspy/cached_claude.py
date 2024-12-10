from typing import Optional, Any
import os
from dsp.modules.anthropic import Claude

class CachingClaude(Claude):
    """Claude wrapper that adds caching headers for Anthropic's prompt caching."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self.client.default_headers.update({
            "anthropic-beta": "prompt-caching-2024-07-31"
        })

    def basic_request(self, prompt: str, **kwargs):
        """Override basic_request to add caching headers"""
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        # caching mechanism requires hashable kwargs
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs.pop("n")
        response = self.client.beta.prompt_caching.messages.create(**kwargs)
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        return response