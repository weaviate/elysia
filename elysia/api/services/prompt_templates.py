import dspy

class TitleCreatorPrompt(dspy.Signature):
    """
    You are an expert at creating a title for a given text.
    """
    text = dspy.InputField(
        description="""
        The text to create a title for.
        """.strip()
    )
    title = dspy.OutputField(
        description="""
        The title for the text. This is a single, short, succinct summary that describes the topic of the conversation.
        """.strip()
    )

class ObjectRelevancePrompt(dspy.Signature):
    """
    You are an expert at determining the relevance of a set of retrieved objects to a user's query.
    """
    user_prompt = dspy.InputField(
        description="""
        The user's input prompt. This is usually a request for information.
        """.strip()
    )
    objects = dspy.InputField(
        description="""
        The set of retrieved objects.
        These are usually a list of dictionaries, with all dictionaries having the same keys.
        You will have to infer the content of these by inspecting the keys and values.
        """.strip()
    )
    any_relevant = dspy.OutputField(
        description="""
        Whether any of the objects are relevant to the user's query. (True/False)
        Return a boolean value, True or False only, with no other text.
        """.strip()
    )