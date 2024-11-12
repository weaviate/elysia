import dspy
from typing import Literal, get_args, Union

def construct_decision_prompt(available_tasks_list: list[str] = None) -> dspy.Signature:
    # Create dynamic Literal type from the list, or use str if None
    TaskLiteral = (Literal[tuple(available_tasks_list)] if available_tasks_list is not None  # type: ignore
                  else str)

    class DecisionPrompt(dspy.Signature):
        """
        You are an expert routing agent, who is responsible for routing a user's prompt to the most appropriate task.
        
        You are a chatbot for the app called Elysia, a chatbot that can retrieve information from a variety of sources and answer questions about it.
        Elysia is an agentic retrieval augmented generation (RAG) service, where users can query from Weaviate collections,
        and you will retrieve the most relevant information and answer the user's question. This includes a variety
        of different ways to query, such as by filtering, sorting, querying multiple collections, and providing summaries
        and textual responses.

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
                    "user": The user's message
                    "assistant": The assistant's response
                }
            ]
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
        # completed_tasks = dspy.InputField(
        #     description="""
        #     A list of tasks that have already been completed.
        #     This is so you know some history of what has been done, so you can avoid repeating tasks, and also so you know what is available to the user, to better decide which property to choose.
        #     This is a list of dicts, where each dict may have different information based on what the task was.
        #     The format is:
        #     - id: the id of the task
        #     - options: a list of the available tasks at the time, with their descriptions as the values
        #     - decision: the name of the task that was decided on at the time
        #     - instruction: the prompt/instruction for this particular decision
        #     - metadata: any metadata that was returned from the task, this is not as important as the other fields
        #     """.strip(),
        #     format = str
        # )
        data_queried = dspy.InputField(
            description="""
            A list of items, showing whether a query has been completed or not.
            This is an itemised list, showing which collections have been queried, and how many items have been retrieved from each.
            If there are 0 items retrieved, then the collection _has_ been queried, but no items were found. Use this in your later judgement.
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
        decision_tree = dspy.InputField(
            description="""
            A full nested dictionary of the entire decision tree.
            This is the _full set_ of tasks available to you _in the future ONLY_, so you should not pick any tasks in this field now.
            This is a dictionary of the format:
            {
                "id": id of the task, semi-descriptive
                "instruction": instruction for the task, very descriptive, detailing what the task is about
                "options": a dictionary of the available options for the task, with the tasks as the keys, the values are dictionaries with the same keys as above,
                e.g.
                {
                    "option_1": {
                        "id": id of the task, semi-descriptive
                        "instruction": instruction for the task, very descriptive, detailing what the task is about
                        "options": {"option_1_1": {...}, "option_1_2": {...}}
                    }
                }
            }
            etc.
            Tasks that depend on other tasks are nested within the options of the current task.
            You should NOT pick any tasks in this field, as it will cause an error.
            Use this field to evaluate what paths you can take in the future. This will help you pick a preliminary task for later reward.
            """.strip(),
            format=str
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