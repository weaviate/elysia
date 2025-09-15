import dspy


class ObjectSummaryPrompt(dspy.Signature):
    """
    Given a list of objects (a list of dictionaries, where each item in the dictionary is a field from the object),
    you must provide a list of strings, where each string is a summary of the object.

    These objects can be of any type, and you should summarise them in a way that is useful to the user.
    """

    objects: list[dict] = dspy.InputField(
        desc="The objects to summarise.", format=list[dict]
    )
    summaries: list[str] = dspy.OutputField(
        desc="""
        The summaries of each individual object, in a list of strings.
        These summaries should be extremely concise, and not aiming to summarise all objects, but a brief summary/description of _each_ object.
        Specifically, what the object is, what it is about to give the user a concise idea of the individual objects. A few sentences at most.
        Your output should be a list of strings in Python format, e.g. `["summary_1", "summary_2", ...]`.
        Do not enclose with ```python or ```. Just the list only.
        """.strip(),
        format=list[str],
    )
