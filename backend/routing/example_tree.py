from typing import Callable, List

import dspy

class Tree:
    def __init__(self, prompt: str, decision_llm: Callable[[str], str]):
        self.prompt = prompt
        self.children = []

    # def decision(self, prompt: str, options: List[str]):


if __name__ == "__main__":

    
    hf_model_name = "meta-llama/Llama-3.2-1B"
    main_model = dspy.HFModel(hf_model_name, hf_device_map="cpu")
    dspy.settings.configure(lm = main_model)

    user_prompt = "Write a summary of all recent github tickets about hot-cold storage separation."