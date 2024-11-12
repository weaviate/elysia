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