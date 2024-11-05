import dspy

class SummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects, summarize the information in the objects to answer the user's prompt.
    """
    user_prompt = dspy.InputField(description="The user's original query")
    available_information = dspy.InputField(
        description="""
        The retrieved objects from the knowledge base.
        This will be in the form of a list of dictionaries, where each dictionary contains the metadata and object fields.
        You should use all of the information available to you to answer the user's prompt, 
        but use judgement to decide which objects are most relevant to the user's query.
        """.strip()
    )
    summary = dspy.OutputField(description="The summary of the retrieved objects")
