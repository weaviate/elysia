import dspy

class SummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects, summarize the information in the objects to answer the user's prompt.
    """
    user_prompt = dspy.InputField(description="The user's original query")
    retrieved_objects = dspy.InputField(description="The retrieved objects from the knowledge base")
    summary = dspy.OutputField(description="The summary of the retrieved objects")
