import dspy
from typing import Literal, get_args, Union

def construct_decision_prompt(available_tasks_list: list[str] = None) -> dspy.Signature:
    # Create dynamic Literal type from the list, or use str if None
    TaskLiteral = (Literal[tuple(available_tasks_list)] if available_tasks_list is not None  # type: ignore
                  else str)

    class DecisionPrompt(dspy.Signature):
        """
        You are an expert routing agent, who is responsible for routing a user's prompt to the most appropriate task.
        You are part of an ensemble of routing agents within Elysia, who each make a decision about which task to complete.
        Your decision is one of many that will be used to make the final decision.

        Given a query (user_prompt) from a user, and a list of possible tasks (available_tasks), decide the task that needs to be completed.
        You should think carefully and logically about the user's input and the tasks that you have available to you (available_tasks), and then decide which task is the most appropriate.
        
        You may be asked to make this decision more than once at a later date, so you should try to choose the most appropriate task NOW, knowing that you may be asked again.

        You are also responsible for checking if the overall goal has been completed, and returning False if it has, and True otherwise.
        Remember, the overall goal is to respond to the user's query in a satisfactory way, and you must do so by completing the necessary tasks.

        To route the user's query to the most appropriate task, follow these steps:
        1. Evaluate what tasks are completed via the previous_reasoning field, as well as the information given within these tasks.
        2. If tasks have already been completed, there may be information that has been retrieved that can help you decide which task to choose.
        3. If there is no information available, then:
            a. You could be in the first decision, in which case you should choose the most appropriate task based on the user's query.
            b. You could be in the process of 
        4. Evaluate whether the user's query can be answered by the available information.
            a. If it can, then you should choose the next most appropriate task.
            b. If it cannot, then you should choose the task so that you can respond to the user's query.
        5. Evaluate whether all the possible actions that can be taken to answer the user have been taken.
            a. If the user's query is not satisfied, then you should choose the next most appropriate task.
            b. If all the actions have been taken, then you should return False for the all_actions_completed field.
        6. Remember that it is possible that the task is impossible, in which case you have done all actions possible, so you should return True for the all_actions_completed field.
        """

        # Regular input fields
        user_prompt = dspy.InputField(
            description="The query that the user is asking"
        )
        instruction = dspy.InputField(
            description="The instruction for the decision. Pay close attention to this. You must choose a task based on this instruction alone.",
            format = str
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

        # Communication-based input fields
        previous_reasoning = dspy.InputField(
            description="""
            Your reasoning that you have output from previous decisions.
            This is so you can use the information from previous decisions to help you make the current decision.
            This should be considered as a train of thought, so you can continue thoughts, build on them, and incorporate them into your current decision.
            Additionally, this means future decisions can also use this information, so it is cumulative, hence your reasoning for this task can include information for future decisions.
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
            Multiple trees represent multiple passes through the tree, so you can use the information from previous passes to help you make the current decision.
            These are sequential, so the decisions in `tree_2` are after the decisions in `tree_1`, and so on.
            DO NOT repeat the same reasoning in the previous_reasoning field, as this will cause a loop.
            You should instead create _different_ reasoning for each decision, so you can build on them.
            Use this to evaluate certain things, such as whether the task is possible given the information available.
            """.strip(),
            format = str
        )

        tree_count = dspy.InputField(
            description="""
            Currently, the number of completed decision trees that you have run through.
            Each tree is a separate pass through the decision tree, so this number informs you of how many times you have gone through the decision tree already.
            Each separate pass through the decision tree has had the opportunity to query collections or respond to the user, so this number informs you of how many attempts have been made to query collections or respond to the user.
            This is a cumulative number out of a total number of recursions (e.g. X/Y), so you know how many attempts have been made in total.
            Use this to evaluate based on how many attempts have been made, and how many more you have left, whether you should continue or not.
            As you approach a larger number of recursions, you should start to consider whether it is possible to continue, as making the same decision over and over is pointless.
            """.strip(),
            format = int
        )
        
        data_queried = dspy.InputField(
            description="""
            A list of items, showing whether a query has been completed or not.
            This is an itemised list, showing which collections have been queried, and how many items have been retrieved from each.
            If there are 0 items retrieved, then the collection _has_ been queried, but no items were found. Use this in your later judgement.
            The information retrieved is in the available_information field.
            """.strip(),
            format = str
        )
        
        current_message = dspy.InputField(
            description="""
            The current message you, the assistant, have written to send to the user. 
            This message has not been sent yet, you will add text to it, to be sent to the user later.
            In essence, the concatenation of this field, current_message, and the text_return field, will be sent to the user.
            """.strip(),
            format = str
        )

        # Task-specific input fields
        available_tasks = dspy.InputField(
            description="""
            A list and description of the tasks that can be completed.
            These are the ONLY tasks that you can choose from to decide which task to complete.
            Do not choose a task that is not in this list.
            Do not assume you know better than the task list. The program is dependent on you choosing an option from this list and ONLY this list.
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
        
        future_information = dspy.InputField(
            description="""
            For each task, what future tasks are available after selecting this action.
            This provides context so you know what you can do in the future if you select each task.
            This is a dictionary of the format:
            {
                "task_1": "description of future tasks after selecting task_1",
                "task_2": "description of future tasks after selecting task_2",
                ...
            }
            Use this to evaluate what paths you can take in the future. This will help you pick a preliminary task for later reward.
            If this is empty, it is likely that this is the last task in a sequence, but it does not necessarily mean this is the last action to take.
            """.strip(),
            format = str
        )

        # Output fields
        task: TaskLiteral = dspy.OutputField(
            description="""
            The decided task. This must be one of the 'name' fields in available_tasks.
            IMPORTANT: This MUST be from the available_tasks list _only_.
            Return the name of the task only, exactly as it appears in the available_tasks list.
            Do not pick a task that is not in the available_tasks list, even if you don't think it is relevant.
            Future tasks will be available for you later, but right now you must choose from the available_tasks list.
            """.strip(),
            format = str
        )
        
        all_actions_completed_reasoning = dspy.OutputField(
            description="""
            Break down all the requests in the user_prompt, and evaluate whether all the possible actions that can be taken to answer the user have been taken.
            To answer this, you should look at the previous_reasoning field, to see if everything that can be done has been done.
            You should also see if there are any other actions that could have been taken in the past that need to be completed.
            """.strip()
        )
        
        all_actions_completed = dspy.OutputField(
            description="""
            _After_ completing the task decided on above, and ONLY this task (as well as the other tasks that have already been completed), 
            will all actions that can be taken to answer the user have been taken?
            Base this on the {{user_prompt}}, the {{instruction}} for the task decided on, and the history of completed tasks.
            It is possible that the task is impossible, in which case you have done all actions possible, so you should return True, as all actions are completed.
            If you identify a task that is in the previous options that hasn't been completed yet, but should be based on the user prompt, then you should return False.
            Otherwise, you should return True.
            Do not include any other information in your response, only a True/False value, and nothing else.
            (True/False)
            """.strip(),
            format=bool
        )
        
        reasoning_update_message = dspy.OutputField(
            desc="""
            Begin this field with the text in current_message field, which is your message _so far_ to the user. Avoid repeating yourself (from the current_message field). 
            If this field is empty, this is a new message you are starting.
            You should write out exactly what it says in current_message, and then afterwards, 
            continue with your new reasoning to communicate anything else to the user specific to the task you have just decided on.
            Your additions should be a brief succint version of the reasoning field, that will be communicated to the user. Do not complete the task within this field, this is just a summary of the reasoning for the decision.
            Communicate this in a friendly and engaging way, as if you are explaining your reasoning to the user in a chat message.
            Do not ask any questions, and do not ask the user to confirm or approve of your actions.
            Your action is _already_ decided, so do not ask the user anything, you are explaining what is already happening.
            If current_message is empty, then this is a new message you are starting, so you should write out only a new message.
            Do NOT attempt to complete the task within this field, or to answer the user's query. You are only communicating your reasoning for the decision in a step-wise fashion. 
            This is displayed to the user as non-primary text, so stick to this brief exactly.
            You should only add one extra sentence to the current_message field, and that is it. Do not add any more.
            Use gender neutral language.
            You should always add an extra sentence to the current_message field, summarising your reasoning and explaining the decision.
            """.strip(),
            format = str
        )
        
        full_chat_response = dspy.OutputField(
            description="""
            The response to the user's prompt. Use gender neutral language.
            Use current_message to frame your response, as if you are continuing the paragraph.
            But this field should be a full response to the user's prompt, so try to answer the user's prompt in full.
            You should still use what information is available to you to answer the user's prompt.
            If nothing is relevant, then you should just respond with a simple text response, apologising that you cannot answer the users query.
            If the recursion limit has been reached, it is likely some or all of the information is not relevant, so you should answer based on what is available.
            If possible, you should apologise for anything you cannot achieve, and suggest alternative ways of prompting that you think would help, based on what you know about the decision process.
            """.strip(),
            format = str
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