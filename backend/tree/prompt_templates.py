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
    instruction = dspy.InputField(
        description="The instruction for the decision"
    )
    completed_tasks = dspy.InputField(
        description="""
        A list of tasks that have already been completed.
        This is so you know some history of what has been done, so you can avoid repeating tasks, and also so you know what is available to the user, to better decide which property to choose.
        This is a list of dicts, where each dict may have different information based on what the task was.
        """.strip()
    )
    available_tasks = dspy.InputField(
        description="""
        A description of the tasks that can be completed.
        These will be in the form of a list of dictionaries, where each dictionary contains two keys:
        - name: The name of the task
        - description: A description of the task
        """.strip()
    )
    task = dspy.OutputField(
        description="""
        The decided task. This must be one of the 'name' fields in available_tasks.
        Your output should be a dictionary with the following keys:
        - name: The name of the task
        - reason: A justification for why this task was chosen
        """.strip()
    )
    user_will_be_satisfied_reasoning = dspy.OutputField(
        description="""
        State your reasoning for why the user will be satisfied after completing the task decided on above.
        """.strip()
    )
    user_will_be_satisfied = dspy.OutputField(
        description="""
        _After_ completing the task decided on above, will the OVERALL GOAL outlined in user_prompt be completed? 
        Will everything that needs to be done to satisfy the user's query be done?
        This includes all parts of what the user is requesting.
        Base this on the {{user_prompt}}, the {{instruction}} for the task decided on, and the history of completed tasks.
        Specifically, if needed, you should check the 'result' field in the completed_tasks list, to see if the requested information has been retrieved or the results of the previous tasks are enough to satisfy the user's query.
        You should also base your decision on the available options that were in the previous tasks, if an option is available that should be explored, then the overall goal is not satisfied.
        If you identify a task that is in the previous options that _needs_ to be completed so that the users prompt is satisfied, then you should return False.
        Otherwise, you should return True.
        (True/False)
        """.strip(),
        format=bool
    )

