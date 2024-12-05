import dspy
from typing import Literal

def construct_training_decision_prompt(available_tasks_list: list[str] = None) -> dspy.Signature:
    # Create dynamic Literal type from the list, or use str if None
    TaskLiteral = (Literal[tuple(available_tasks_list)] if available_tasks_list is not None  # type: ignore
                  else str)



    class TrainingDecisionPrompt(dspy.Signature):
        """
        You are a copy of a routing agent within Elysia, responsible for selecting the most appropriate next task to handle a user's query.
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

        However, the task has already been decided, and your goal is to act like the agent that made the decision, and output the reasoning for that decision, as if you were making that decision for the first time.
        """

        # Regular input fields

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
            - Note: 0 items means query executed but found nothing
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
        task: TaskLiteral = dspy.InputField(
            description="""
            The decided task. 
            """.strip(),
            format = str
        )
        reasoning_update_message: str = dspy.OutputField(
            description="Write out current_message in full, then add one sentence to the paragraph which explains your task selection logic. Mark your new sentence with <NEW></NEW>. If current_message is empty, your whole message should be enclosed in <NEW></NEW>. Use gender-neutral language and communicate to the user in a friendly way."
        )
        full_chat_response: str = dspy.OutputField(
            description="Complete response to user based on available information. If the user cannot be satisfied, explain why and suggest alternative approaches."
        )
        reasoning = dspy.OutputField(
            description="""
            The reasoning for the decision of the task.
            You should output this reasoning _as if you are deciding on it_. Pretend it has not already been decided, and that you are an agent making this decision and choosing this task.
            """.strip(),
            format = str
        )

    return TrainingDecisionPrompt
