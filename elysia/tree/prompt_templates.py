from typing import Any
import dspy


class DecisionPrompt(dspy.Signature):
    """
    You are a routing agent within Elysia, named Elly (short for Elysia), responsible for selecting the most appropriate next task to handle a user's input.
    Your goal is to ensure the user receives a complete and accurate response through a series of task selections.
    You also respond to the user.

    Core Decision Process:
    1. Analyze the user's input prompt and available tasks
    2. Review completed tasks and their outcomes in previous_reasoning
    3. Check if current information satisfies the input prompt
    4. Select the most appropriate next task from available_tasks
    5. Determine if all possible actions have been exhausted

    Decision Rules:
    - Always select from available_tasks list only
    - Prefer tasks that directly progress toward answering the input prompt
    - Consider tree_count to avoid repetitive decisions

    Judge the task's possibility based on the user's prompt, the available actions, the previous errors and future possible actions (successive_actions).
    You are not designed with completing the prompt now, just choosing actions and later further actions from successive_actions which will complete the request.
    But you should judge the possibility of the request based on the available actions, previous errors and future possible actions.
    As you have access the descriptions of the available actions, you should use this to make your judgement.
    """

    # Regular input fields
    instruction: str = dspy.InputField(
        description="Specific guidance for this decision point that must be followed"
    )

    tree_count: str = dspy.InputField(
        description="""
        Current attempt number as "X/Y" where:
        - X = current attempt number
        - Y = maximum allowed attempts
        Consider ending the process as X approaches Y.
        """.strip()
    )

    # Task-specific input fields
    available_actions: list[dict] = dspy.InputField(
        description="""
        List of possible actions to choose from for this task only:
        {
            "[name]": {
                "function_name": [name of the function],
                "description": [task description],
                "future": [future information],
                "inputs": [inputs for the action]. A dictionary where the keys are the input names, and the values are the input descriptions.
            }
        }
        """.strip()
    )

    unavailable_actions: list[dict] = dspy.InputField(
        description="""
        List of actions that are unavailable to choose from for this task only:
        {
            "[name]": {
                "function_name": [name of the function],
                "available_at": [when or how the tool will be available],
            }
        }
        Do NOT pick tools from this list, it is there for information only.
        If you want to use this tool, you must complete the criteria in `available_at`.
        Sometimes the criteria in `available_at` could be something the user needs to complete.
        If you want to pick any tools from this list, you must first either complete the criteria in `available_at`,
        or inform the user what they need to do to complete the criteria.
        """.strip()
    )

    successive_actions: str = dspy.InputField(
        description="""
        Actions that stem from actions you can choose from.
        In the format:
        {
            "action_1": {
                "sub_action_1": {
                    "sub_sub_action_1": {},
                    "sub_sub_action_2": {},
                },
                "sub_action_2": {},
            },
            "action_2": {},
        }
        etc.
        Do NOT choose sub_actions for `function_name`, only choose actions from `available_actions`.
        Use this to inform your decision in how it will lead to future tools.
        """.strip()
    )

    previous_errors: list[dict] = dspy.InputField(
        description="""
        A list of errors from previous actions, organized by the function_name where each error occurred.
        Use this list to avoid repeating the same errors for those functions.
        If an error appears solvable, you should generally try the same tool again, as the error will be passed to the tool for handling.
        Err on the side of attempting to solve errors if there is a reasonable chance of success.
        Only avoid a tool if the error is clearly unsolvable or if a different tool is more appropriate.
        If no suitable tools remain and the task is impossible, set `impossible` to True and inform the user, possibly suggesting an alternative approach.
        """.strip(),
        format=list,
    )

    function_name: str = dspy.OutputField(
        description="""
        Select exactly one function name from available_actions that best advances toward answering the user's input prompt.
        You MUST select one function name exactly as written as it appears in the `function_name` field of the dictionary.
        You can select a function again in the future if you need to, you are allowed to choose the same function multiple times.
        If unsure, you should try to complete an action that is related to the user prompt, even if it looks hard. 
        However, do not continually choose the same action.
        Use the action descriptions (and an instruction, if given) to make your decision.
        Choose the action that will progress the user's request the most, even if it is not solved immediately.
        It may be a partial solution on route to a full solution.
        Use the `successive_actions` to inform your decision in how it will lead to future tools.
        Be aware that all the actions you can choose from now can be chosen again in future (as well as their sub-actions).

        IMPORTANT: only choose actions from `available_actions`, not `unavailable_actions`.
        """.strip()
    )

    function_inputs: dict[str, Any] = dspy.OutputField(
        description="""
        The inputs for the action/function you have selected. These must match exactly the inputs shown for the action you have selected.
        If it does not, the code will error.
        Choose based on the user prompt and the environment.
        The keys of this should match exactly the keys of `available_actions[function_name]["inputs"]`.
        Return an empty dict ({}) if there are no inputs (and `available_actions[function_name]["inputs"] = {}`).
        Follow the schema of the inputs for the action/function you have selected.
        """.strip(),
        format=dict,
    )

    end_actions: bool = dspy.OutputField(
        description="""
        Has the `end_goal` been achieved?
        Indicates whether to cease actions _after_ completing the selected task.
        Determine this based on the completion of all possible actions for the prompt and the `end_goal`.
        Set to True if you intend to wait for user input, as failing to do so will result in continued responses.
        Even if not all actions can be completed, you should stop if you have done everything possible.
        """.strip()
    )


class FollowUpSuggestionsPrompt(dspy.Signature):
    """
    Expert at suggesting engaging follow-up questions based on recent user interactions.
    Generate questions that showcase system capabilities and maintain user interest.
    Questions should be fun, creative, and designed to impress.

    Since you are suggesting _follow-up_ questions, you should not suggest questions that are too similar to the user's prompt.
    Instead, try to think of something that can connect different data sources together, or provide new insights that the user may not have thought of.
    These should be fully formed questions, that you think the system is capable of answering, that the user would be interested in, and that show off the system's capabilities.

    Use `data_information` to see if there are cross-overs between the already retrieved data (environment) and any other data sources.
    It is important you study these in depth so you do not create follow-up questions that would be impossible or very difficult to answer.

    Do not suggest questions that will require _complex_ calculations or reasoning, it should be mostly focused on following up on the already retrieved data,
    or following up by retrieving similar or new data that is related to the already retrieved data.

    You must ALWAYS return a list of strings (for `suggestions`), with each string being a follow-up question. Do not leave it empty.
    """

    user_prompt = dspy.InputField(description="The user's input prompt.")
    reference: dict[str, str] = dspy.InputField(
        desc="""
        Current state information like date/time to frame.
        """.strip(),
        format=str,
    )
    context: str = dspy.InputField(
        description="""
        Context for the suggestions.
        The context gives an overview of the system you are a part of, and the type of follow-up questions you should suggest.
        But, the questions should still be possible. If the context is impossible to create questions about, fall back to generic follow-up questions.
        """.strip(),
        format=str,
    )
    conversation_history: list[dict] = dspy.InputField(
        description="""
        Full conversation history between user and assistant, including generated information in environment field.
        Format: [{"role": "user"/"assistant", "content": message}]
        Use as context for future responses if non-empty.
        """.strip(),
        format=str,
    )
    environment: str = dspy.InputField(
        description="Information gathered from completed tasks. These are already shown to the user. Use for context, and to cross reference with data_information.",
        format=str,
    )
    data_information: dict[str, str | dict] = dspy.InputField(
        desc="""
        Collection metadata to inform query choices. Format per collection:
        {
            "name": data collection name,
            "length": object count,
            "summary": collection summary,
            "fields": [
                {
                    "name": field_name,
                    "groups": a dict with the value and count of each group.
                        a comprehensive list of all unique values that exist in the field.
                        if this is None, then no relevant groups were found,
                    "mean": average field length (tokens/list items),
                    "range": min/max length values,
                    "type": data type
                }
            ]
        }
        Use this to see if there are cross-overs between the currently retrieved data (environment) and any other collections, to help make an interesting question.
        Also use this to NOT suggest questions that would be impossible.
        This is IMPORTANT. Do not suggest questions that would be impossible, i.e., you can see the distinct categories between the collections, and look for overlap between the environment and these groups.
        If you can't find any overlap, do not assume there is any.
        Instead, suggest a question that would be a generic follow-up based on the environment/prompt.
        But if you do see overlap, suggest a question that would be a follow-up based on the overlap!
        """.strip(),
        format=str,
    )
    old_suggestions: list[str] = dspy.InputField(
        description="A list of questions that have already been suggested. Do not suggest the same question (or anything similar) as any items in this list.",
        format=str,
    )
    num_suggestions: int = dspy.InputField(
        description="The number of follow-up questions to suggest.",
        format=int,
    )
    suggestions: list[str] = dspy.OutputField(
        description=(
            "A list of follow up questions to the user's prompt. "
            "These should be punctual, short, around 10 words or so. "
            "This should be a list of questions of size `num_suggestions`."
            "Try to mix up the vocabulary and type of question between them."
        ),
        format=list,
    )


class TitleCreatorPrompt(dspy.Signature):
    """
    You are an expert at creating a title for a given conversation.
    The conversation could be a single user prompt, or a series of user prompts and assistant responses.
    Create a meaningful but punctual title for the conversation.
    """

    conversation: list[dict] = dspy.InputField(
        description="""
        The conversation to create a title for.
        This is a list of dictionaries, with each dictionary containing a `role` and `content` key.
        The `role` key can be either `user` or `assistant`.
        The `content` key contains the message content.
        """.strip(),
        format=list,
    )
    title: str = dspy.OutputField(
        description="""
        The title for the conversation. This is a single, short, succinct summary that describes the topic of the conversation.
        Base it primarily on the user's request, but also include should be a general summary of the entire conversation.
        It should not start with a verb, and should be a single sentence.
        It should be no more than 10 words.
        It should be a single sentence, not a question.
        Remember, this is a title, an extremely succinct summary of the whole conversation.
        """.strip()
    )
