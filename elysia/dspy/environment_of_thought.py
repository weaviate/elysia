import dspy
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature

class EnvironmentOfThought(Module):
    def __init__(self, signature, activated=True, **config):
        super().__init__()

        self.activated = activated

        self.signature = signature = ensure_signature(signature)
        *_keys, last_key = signature.output_fields.keys()

        observation_prefix = "Observation: Break down the environment into smaller, relevant parts, and record your observations in how it relates to the query in order to help us reason"
        context_prefix = "Context: Reproduce relevant parts of the environment that are most relevant to the task"
        reasoning_prefix = "Reasoning: Using knowledge from the context and your observations, think step by step in order to answer the query"
        
        observation_desc = "${observation}"
        context_desc = "${context}"
        reasoning_desc = "${reasoning}"

        observation_field = dspy.OutputField(prefix=observation_prefix, desc=observation_desc)
        context_field = dspy.OutputField(prefix=context_prefix, desc=context_desc)
        reasoning_field = dspy.OutputField(prefix=reasoning_prefix, desc=reasoning_desc)
        
        extended_signature = signature.prepend("observation", observation_field, type_=str)
        extended_signature = extended_signature.prepend("context", context_field, type_=str)
        extended_signature = extended_signature.prepend("reasoning", reasoning_field, type_=str)

        self._predict = dspy.Predict(extended_signature, **config)
        self._predict.extended_signature = extended_signature

    def forward(self, **kwargs):
        assert self.activated in [True, False]

        signature = kwargs.pop("new_signature", self._predict.extended_signature if self.activated else self.signature)
        return self._predict(signature=signature, **kwargs)

    @property
    def demos(self):
        return self._predict.demos

    @property
    def extended_signature(self):
        return self._predict.extended_signature
