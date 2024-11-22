import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

base_lm = dspy.LM(model="claude-3-5-haiku-20241022", max_tokens=8000)
complex_lm = dspy.LM(model="claude-3-5-sonnet-20241022", max_tokens=8000)

dspy.settings.configure(lm=base_lm)