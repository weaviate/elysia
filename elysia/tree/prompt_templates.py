import dspy

class DecisionPrompt(dspy.Signature):
    """
    You are an expert routing agent, who is responsible for routing a user's prompt to the most appropriate task.
    Given a query (user_prompt) from a user, decide the task that needs to be completed.
    You should think carefully and logically about the user's input and the tasks that you have available to you, and then decide which task is the most appropriate.
    
    You may be asked to make this decision more than once at a later date, so you should try to choose the most appropriate task, knowing that you may be asked again.
    Therefore you can pick the task that _starts_ the process of solving the user's query, even if there are other tasks that may be more appropriate immediately.

    Your ultimate goal is to respond to the user in a satisfying way, such as producing reasonable chat output if they ask a question,
    or producing a summary of the information if they want a high-level overview. Sometimes, they may only want to display information,
    in which case you should choose the task that best does this.
    It is up to you to decide what is most appropriate, but you should think about the user's query and the tasks available to you.

    You are also responsible for checking if the overall goal has been completed, and returning False if it has, and True otherwise.
    Remember, the overall goal is to satisfy the user's query, and you must do so by completing the necessary tasks.
    """

    user_prompt = dspy.InputField(
        description="The query that the user is asking"
    )
    reference = dspy.InputField(
        description="Information about the state of the world NOW such as the date and time, used to frame the decision making.",
        format = str
    )
    conversation_history = dspy.InputField(
        description="""
        The conversation history between the user and the assistant (you), including all previous messages.
        During this conversation, the assistant has also generated some information, which is also relevant to the decision.
        This information is stored in `available_information` field.
        If this is non-empty, then you have already been speaking to the user, and these were your responses, so future responses can use these as context.
        The history is a list of dictionaries of the format:
        [
            {
                "user": The user's message
                "assistant": The assistant's response
            }
        ]
        """.strip(),
        format = str
    )
    instruction = dspy.InputField(
        description="The instruction for the decision"
    )
    completed_tasks = dspy.InputField(
        description="""
        A list of tasks that have already been completed.
        This is so you know some history of what has been done, so you can avoid repeating tasks, and also so you know what is available to the user, to better decide which property to choose.
        This is a list of dicts, where each dict may have different information based on what the task was.
        The format is:
        - id: the id of the task
        - options: a list of the available tasks at the time, with their descriptions as the values
        - decision: the name of the task that was decided on at the time
        - instruction: the prompt/instruction for this particular decision
        - metadata: any metadata that was returned from the task, this is not as important as the other fields
        """.strip(),
        format = str
    )
    available_tasks = dspy.InputField(
        description="""
        A list and description of the tasks that can be completed.
        This is what you will choose from to decide which task to complete.
        To learn about the tasks, you should look at the 'description' field in the corresponding dictionary entry for each task.
        These will be in the form of a list of dictionaries, where each dictionary contains values:
        - description: A description of the task (this is for your information and to help you decide)
        - action: The function that will be called to complete the task, not relevant to your decision
        - returns: The type of object that will be returned from the task, not relevant to your decision
        - next: The next task to be completed, not relevant to your decision
        """.strip(),
        format = str
    )
    available_information = dspy.InputField(
        description="""
        A list of information that is available to the user, based on the history of completed tasks.
        This is likely a list of retrieved objects, or summaries, or other information.
        You should use this information to judge whether more information is needed to satisfy the user's query.
        If this is empty, then no information has currently been retrieved, but it can be, depending on the options available to you.
        """.strip(),
        format = str
    )
    task = dspy.OutputField(
        description="""
        The decided task. This must be one of the 'name' fields in available_tasks.
        IMPORTANT: This MUST be from the available_tasks list _only_.
        Your output should be a dictionary that is a proper JSON object (i.e. enclosing with double quotes) with the following keys:
        - name: The name of the task
        - reason: A justification for why this task was chosen
        """.strip()
    )
    possible_future_tasks = dspy.InputField(
        description="""
        A list of all the tasks that could be completed after the current task is completed.
        This is the _full set_ of tasks available to you in the future, past and present.
        Some of these may have already been completed, which you can find in the {{completed_tasks}} field.
        This is a dictionary of the format:
        {
            "task1": {"task1_1": {"task1_1_1": {}}, "task1_2": {"task1_2_1": {}}},
            "task2": {"task2_1": {"task2_1_1": {}}, "task2_2": {"task2_2_1": {}}}
        }
        etc.
        Tasks that depend on other tasks are nested within them.
        Use this to evaluate whether the user will be satisfied after completing the current task.
        """.strip(),
        format=str
    )
    user_will_be_satisfied_reasoning = dspy.OutputField(
        description="""
        Break down all the requests in the user_prompt into smaller parts in an itemised list.
        After completing the currently decided task, will all the possible actions that can be taken to satisfy the user's query have been taken?
        To answer this, you should look at the {{available_information}} field, to see if the information that is available satisfies the user's query.
        You should also take into account the history of completed tasks in {{completed_tasks}}, to see if there are any other actions that could have been taken in the past that need to be completed.
        Output a single boolean value, indicating whether the user will be satisfied as a whole (all parts of the user's query are satisfied) after completing the currently decided task.
        """.strip()
    )
    user_will_be_satisfied = dspy.OutputField(
        description="""
        _After_ completing the task decided on above, and ONLY this task (as well as the other tasks that have already been completed), will the OVERALL GOAL outlined in user_prompt be completed? 
        Have all the _actions_ that _can be taken_, been taken to attempt to satisfy the user's query?
        Do not include any other information in your response, only a True/False value, and nothing else.
        This includes all parts of what the user is requesting, so break this down into smaller parts if needed.
        Base this on the {{user_prompt}}, the {{instruction}} for the task decided on, and the history of completed tasks.
        Specifically, if needed, you should check the 'result' field in the completed_tasks list, to see if the requested information has been retrieved or the results of the previous tasks are enough to satisfy the user's query.
        You should also base your decision on the available options that were in the previous tasks, if an option is available that should be explored, then the overall goal is not satisfied.
        If you identify a task that is in the previous options that _needs_ to be completed so that the users prompt is satisfied, then you should return False.
        Otherwise, you should return True.
        (True/False)
        """.strip(),
        format=bool
    )


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