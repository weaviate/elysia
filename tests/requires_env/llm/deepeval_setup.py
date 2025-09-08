import os

import torch
import transformers
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(override=True)


class CustomLlama(DeepEvalBaseLLM):
    def __init__(self, max_length: int = 5000):
        self.max_length = max_length

        model_4bit = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=os.getenv("HF_TOKEN"),
        )

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=self.max_length,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"


# if __name__ == "__main__":
#     llm = CustomLlama()
#     print(llm.generate("Hello, how are you?"))

#     from deepeval.metrics import AnswerRelevancyMetric

#     metric = AnswerRelevancyMetric(model=llm)
#     metric.measure(
#         model=llm,
#         data=data,
#     )
