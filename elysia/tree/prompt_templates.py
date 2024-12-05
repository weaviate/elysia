import dspy
from typing import Literal, get_args, Union

def construct_decision_prompt(available_tasks_list: list[str] = None) -> dspy.Signature:
    # Create dynamic Literal type from the list, or use str if None
    TaskLiteral = (Literal[tuple(available_tasks_list)] if available_tasks_list is not None  # type: ignore
                  else str)

    class DecisionPrompt(dspy.Signature):
        """
        You are a routing agent within Elysia, responsible for selecting the most appropriate next task to handle a user's query.
        Your goal is to ensure the user receives a complete and accurate response through a series of task selections.

        Core Decision Process:
        1. Analyze the user's query and available tasks
        2. Review completed tasks and their outcomes in previous_reasoning
        3. Check if current information satisfies the query
        4. Select the most appropriate next task from available_tasks
        5. Determine if all possible actions have been exhausted

        Decision Rules:
        - Always select from available_tasks list only
        - Prefer tasks that directly progress toward answering the query
        - Mark all_actions_completed=True only when:
          * All relevant tasks have been completed successfully, OR
          * The query is impossible to satisfy with available tasks
        - Consider tree_count to avoid repetitive decisions
        """


        # Regular input fields
        user_prompt: str = dspy.InputField(
            description="The user's original query that needs to be answered"
        )
        instruction: str = dspy.InputField(
            description="Specific guidance for this decision point that must be followed"
        )
        reference: dict = dspy.InputField(
            description="Current context information (e.g., date, time) for decision-making"
        )
        conversation_history: list[dict] = dspy.InputField(
            description="""
            Previous messages between user and assistant in chronological order:
            [{"role": "user"|"assistant", "content": str}]
            Use this to maintain conversation context and avoid repetition.
            """.strip()
        )

        # Collection information for user to ask basic questions about collection
        collection_information: dict = dspy.InputField(
            description="""
            Metadata about available collections:
            {
                "name": str,
                "length": int,
                "summary": str,
                "fields": {
                    "field_name": {
                        "groups": list[str],  # unique values for text fields
                        "mean": float,        # average length/value
                        "range": [min, max],
                        "type": str
                    }
                }
            }
            Use to determine if user's request is possible with available data.
            """.strip()
        )
        # Communication-based input fields
        previous_reasoning: dict = dspy.InputField(
            description="""
            Your previous decision logic across multiple attempts:
            {
                "tree_1": {"decision_1": str, "decision_2": str},
                "tree_2": {"decision_1": str, "decision_2": str}
            }
            Use to build on past decisions and avoid repeating failed approaches.
            Each tree represents a complete attempt at answering the query.
            """.strip()
        )

        tree_count: str = dspy.InputField(
            description="""
            Current attempt number as "X/Y" where:
            - X = current attempt number
            - Y = maximum allowed attempts
            Consider ending the process as X approaches Y.
            """.strip()
        )
        
        data_queried: str = dspy.InputField(
            description="""
            Record of collection queries and results:
            - Which collections were searched
            - Number of items retrieved per collection
            Use this to determine whether future searches for this prompt are necessary.
            If the search has been performed multiple times, it is unlikely performing the same search again will retrieve any more information.
            However, if the prompt is for a nested task, you may need to perform the search multiple times to retrieve all relevant information.
            """.strip()
        )
        
        current_message: str = dspy.InputField(
            description="""
            Partial response being built for the user.
            Your additions will be appended to this message.
            """.strip()
        )

        # Task-specific input fields
        available_tasks: list[dict] = dspy.InputField(
            description="""
            List of possible tasks to choose from:
            {
                "[name]": [task description]
            }
            You MUST select one task name exactly as written as it appears in the keys of the dictionary.
            """.strip()
        )
        
        available_information: str = dspy.InputField(
            description="""
            Information gathered from completed tasks.
            Empty if no data has been retrieved yet.
            Use to determine if more information is needed.
            """.strip()
        )
        
        future_information: dict = dspy.InputField(
            description="""
            Available follow-up tasks for each current task choice:
            {"task_name": "description of subsequent possible tasks"}
            Use to plan multi-step approaches to answering the query.
            """.strip()
        )

        # Output fields
        task: TaskLiteral = dspy.OutputField(
            description="Select exactly one task name from available_tasks that best advances toward answering the user's query."
        )
        
        all_actions_completed_reasoning: str = dspy.OutputField(
            description="""
            Break down all the requests in the user_prompt, and evaluate whether all the possible actions that can be taken to answer the user have been taken.
            To answer this, you should look at the previous_reasoning field, to see if everything that can be done has been done.
            You should also see if there are any other actions that could have been taken in the past that need to be completed.
            """.strip()
        )

        all_actions_completed: bool = dspy.OutputField(
            description="""
            - True: Choose this if (1) all necessary information to answer the query is available, or (2) the query cannot be answered with the available tasks.
            - False: Choose this if additional tasks might help answer the query.

            Be pragmatic:

            - Pick True when retrieved information relates to the prompt, even if you cannot answer the query directly.
            - Do not assess the usefulness of the information, only its relevance to the prompt.

            If nested tasks are needed (e.g., multiple queries/multiple collections), pick False until all relevant information is retrieved. Then switch to True.
            """.strip()
        )
        
        reasoning_update_message: str = dspy.OutputField(
            description="Write out current_message in full, then add one sentence to the paragraph which explains your task selection logic. Mark your new sentence with <NEW></NEW>. If current_message is empty, your whole message should be enclosed in <NEW></NEW>. Use gender-neutral language and communicate to the user in a friendly way."
        )
        
        full_chat_response: str = dspy.OutputField(
            description="Complete response to user based on available information. If the user cannot be satisfied, explain why and suggest alternative approaches."
        )

    return DecisionPrompt


class InputPrompt(dspy.Signature):
    """
    You are an expert at breaking down a global task, which is a set of instructions, into multiple smaller distinct subtasks.
    IMPORTANT: Only include the parts that are necessary to complete the task, do not add redundant information related to style, tone, etc.
    These should only be parts of the instructions that can map to _actual_ tasks.
    """
    task = dspy.InputField(
        description="""
        The overall task to break down. 
        This is a set of instructions provided by the user that will be completed later.
        These are usually in conversational or informal language, and are not necessarily task-like.
        """.strip()
    )
    subtasks = dspy.OutputField(
        description="""
        The breakdown of the overall task into smaller distinct subtasks.
        These should be in task-like language, and be as distinct as possible.
        Your responses to this field should not include anything other than what was requested in the task field, but broken down into smaller parts with more descriptive instructions.
        """.strip()
    )