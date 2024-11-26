import dspy

class SummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects, summarize the information in the objects to answer the user's prompt.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to summarize the information in the retrieved objects to answer the user's prompt.
    """
    user_prompt = dspy.InputField(description="The user's original query")
    reference = dspy.InputField(
        description="Information about the state of the world NOW such as the date and time, used to frame the summarization.",
        format = str
    )
    previous_reasoning = dspy.InputField(
        description="""
        Your reasoning that you have output from previous decisions.
        This is so you can use the information from previous decisions to help you respond to the user's prompt.
        This is a dictionary of the form:
        {
            "tree_1": 
            {
                "decision_1": "Your reasoning for the decision 1",
                "decision_2": "Your reasoning for the decision 2",
                ...
            },
            "tree_2": {
                "decision_1": "Your reasoning for the decision 1",
                "decision_2": "Your reasoning for the decision 2",
                ...
            }
        }
        where `tree_1`, `tree_2`, etc. are the ids of the trees in the tree, and `decision_1`, `decision_2`, etc. are the ids of the decisions in the tree.
        This may help you, as previous decisions might give insight into what to say.
        """.strip(),
        format = str
    )
    conversation_history = dspy.InputField(
        description="""
        The conversation history between the user and the assistant (you), including all previous messages.
        During this conversation, the assistant has also generated some information, which is also relevant to the decision.
        This information is stored in `available_information` field.
        If this is non-empty, then you have already been speaking to the user, and these were your responses, so future responses should use these as context.
        The history is a list of dictionaries of the format:
        [
            {
                "role": "user" or "assistant",
                "content": The message
            }
        ]
        In the order which the messages were sent.
        """.strip(),
        format = str
    )
    available_information = dspy.InputField(
        description="""
        The retrieved objects from the knowledge base.
        This will be in the form of a list of dictionaries, where each dictionary contains the metadata and object fields.
        You should use all of the information available to you to answer the user's prompt, 
        but use judgement to decide which objects are most relevant to the user's query.
        """.strip()
    )
    subtitle = dspy.OutputField(description="A subtitle for the summary")
    summary = dspy.OutputField(description="""
    The summary of the retrieved objects. You can use markdown formatting.
    Don't provide an itemised list of the objects, since they will be displayed to the user anyway.
    Your summary should take account what the user prompt is, and the information in the retrieved objects,
    and be a natural continuation of the conversation history, whilst summarising the information.
    """.strip())

class TextResponsePrompt(dspy.Signature):
    """
    You are a helpful assistant, designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    Given a user_prompt, as well as a list of retrieved objects, respond to the user's prompt.
    Your response should be informal, polite, and assistant-like.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to respond to the user's query.
    """
    user_prompt = dspy.InputField(description="The user's original query")
    reference = dspy.InputField(
        description="Information about the state of the world NOW such as the date and time, used to frame the response.",
        format = str
    )
    previous_reasoning = dspy.InputField(
        description="""
        Your reasoning that you have output from previous decisions.
        This is so you can use the information from previous decisions to help you respond to the user's prompt.
        This is a dictionary of the form:
        {
            "tree_1": 
            {
                "decision_1": "Your reasoning for the decision 1",
                "decision_2": "Your reasoning for the decision 2",
                ...
            },
            "tree_2": {
                "decision_1": "Your reasoning for the decision 1",
                "decision_2": "Your reasoning for the decision 2",
                ...
            }
        }
        where `tree_1`, `tree_2`, etc. are the ids of the trees in the tree, and `decision_1`, `decision_2`, etc. are the ids of the decisions in the tree.
        This may help you, as previous decisions might give insight into what to say.
        """.strip(),
        format = str
    )
    conversation_history = dspy.InputField(
        description="""
        The conversation history between the user and the assistant (you), including all previous messages.
        During this conversation, the assistant has also generated some information, which is also relevant to the decision.
        This information is stored in `available_information` field.
        If this is non-empty, then you have already been speaking to the user, and these were your responses, so future responses should use these as context.
        The history is a list of dictionaries of the format:
        [
            {
                "role": "user" or "assistant",
                "content": The message
            }
        ]
        In the order which the messages were sent.
        """.strip(),
        format = str
    )
    available_information = dspy.InputField(
        description="""
        The retrieved objects from the knowledge base.
        This will be in the form of a list of dictionaries, where each dictionary contains the metadata and object fields.
        You should use all of the information available to you to answer the user's prompt, 
        but use judgement to decide which objects are most relevant to the user's query.
        """.strip()
    )    
    current_message = dspy.InputField(
        description="""
        The current message you, the assistant, have written to send to the user. 
        This message has not been sent yet, you will add text to it, to be sent to the user later.
        In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
        """.strip(),
        format = str
    )
    response = dspy.OutputField(
        description="""
        The response to the user's prompt. This is a continuation of the current_message field. 
        This response should be a natural continuation of the current_message field, as if you are continuing the paragraph.
        Use present tense in your text, as if you are currently completing the action.
        If the current_message field is empty, then this response is the beginning of a new message.
        Use gender neutral language.
        """.strip()
    )