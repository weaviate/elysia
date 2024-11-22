import datetime
import random
import os
import dspy
import sys
import json

sys.path.append(os.getcwd())
os.chdir("../..")

from dspy.teleprompt import LabeledFewShot, BootstrapFewShotWithRandomSearch

from elysia.tree.objects import Returns
from elysia.util.parsing import format_datetime
from elysia.tree.tree import Tree
from elysia.tree.prompt_executors import DecisionExecutor

def remove_whitespace(text: str) -> str:
    return " ".join(text.split())

def load_example_from_dict(d: dict):
    return dspy.Example({k: v for k, v in d.items()}).with_inputs(
        "user_prompt", 
        "reference", 
        "conversation_history", 
        "available_information", 
        "instruction", 
        "completed_tasks", 
        "available_tasks", 
        "data_queried",
        "possible_future_tasks",
        "previous_reasoning"
    )

def create_example(
        user_prompt: str,  # user input
        run_num_trees: int, # number of times to run the tree
        run_until_node_id: str, # the next decision to be made in the tree
        reasoning: str, # output field: reasoning for the decision
        task: str, # output field: the decided task
        all_actions_completed_reasoning: str, # output field: reasoning for whether the user will be satisfied
        all_actions_completed: bool, # output field: whether the user will be satisfied
        possible_tasks: list[str], # metric field: the possible tasks that can be completed , worth equal points
        # tasks_to_complete: list[str], # metric field: the tasks that need to be completed
    ):
    """
    Create a decision example.
    This is adaptive, sometimes, we will need to run the decision a few times so that there is a 'history' of previous decisions.
    And then evaluate based on the _next_ one.
    """

    # get tree object
    tree = Tree(verbosity=0, run_num_trees=run_num_trees, run_until_node_id=run_until_node_id)

    # random date for reference field
    year = random.randint(2020, 2024)
    month = random.randint(1, 12)

    date = datetime.datetime(year, month, random.randint(1, 28))
    reference = {
        "datetime": format_datetime(date),
        "day_of_week": date.strftime("%A"),
        "time_of_day": date.strftime("%I:%M %p")
    }


    # run the tree
    results = tree.process_sync(user_prompt)

    # get the variables for the next node
    conversation_history = tree.conversation_history
    available_information = tree.returns.to_json()
    instruction = tree.decision_nodes[run_until_node_id].instruction
    completed_tasks = tree.previous_info
    available_tasks = tree.decision_nodes[run_until_node_id].options
    data_queried = tree.data_queried
    decision_tree = tree.tree
    previous_reasoning = tree.previous_reasoning

    print(f"Created example for '{user_prompt}'")
    print(f"History of tasks:")
    for decision_id in tree.decision_history:
        for info in tree.previous_info:
            if info["id"] == decision_id:
                print(f"    - {decision_id}: {info['decision']}")


    # return the example as a dspy.Example object
    return dspy.Example(
        user_prompt=user_prompt,
        reference=reference,
        reasoning=remove_whitespace(reasoning),
        task=task,
        possible_tasks=possible_tasks,
        all_actions_completed_reasoning=remove_whitespace(all_actions_completed_reasoning),
        all_actions_completed=all_actions_completed,
        conversation_history=conversation_history,
        available_information=available_information,
        instruction=instruction,
        completed_tasks=completed_tasks,
        available_tasks=available_tasks,
        data_queried=data_queried,
        decision_tree=decision_tree,
        previous_reasoning=previous_reasoning
    ).with_inputs(
        "user_prompt", 
        "reference", 
        "conversation_history", 
        "available_information", 
        "instruction", 
        "completed_tasks", 
        "available_tasks", 
        "data_queried",
        "decision_tree",
        "previous_reasoning"
    )


# def metric(example: dspy.Example, pred: dspy.Prediction, trace=None, verbose=False):
#     completed_tasks = [task['id'] for task in pred.completed_tasks]
    
#     if verbose:
#         print(f"Tasks to complete: {example.tasks_to_complete}")
#         print(f"Tree completed tasks: {completed_tasks}")

#     out = 0
#     for task in example.tasks_to_complete:
#         if task in completed_tasks:
#             out += 1

#     return out/len(example.tasks_to_complete)

def metric(example: dspy.Example, pred: dspy.Prediction, trace=None, verbose=False):
    prediction, _ = pred
    return (example.task == prediction.task) or (prediction.task in example.possible_tasks)

# DATA: where the tree has nodes as follows
# TREE: 1. base: - query
#                   - summarize
#                   - text_response
#       2. collection: - example_verba_email_chains
#                      - example_verba_slack_conversations
#                      - example_verba_github_issues
#       3. conversation_choice: - messages_only
#                               - full_conversation
def make_examples():

    # save the examples
    filepath = os.path.join(
        "elysia",
        "training",
        "data"
    )
        
    filename = "decision_examples_example_tree.json"

    if not os.path.exists(os.path.join(filepath, filename)):
        data = [
            create_example(
                user_prompt = "summarize messages from Kaladin", 
                run_until_node_id = "conversation_choice",
                run_num_trees = 1,
                reasoning = """
                Since we are looking at messages from Kaladin, I should return messages_only. 
                The user is looking for a summary of only Kaladin's messages, so there is no need to include information 
                from messages that Kaladin has not written.
                """,
                task = "messages_only",
                all_actions_completed_reasoning = """
                The user will not be satisfied, because I have not yet provided a summary.
                The tasks completed so far only retrieve information, they do not provide a summary.
                In the future, I will need to write a summary, but right now they are not satisfied.
                Since the summary is an additional action I can perform, the user will not be satisfied.
                """,
                possible_tasks = ["messages_only"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "What was Vin discussing in her most recent 10 emails?", 
                run_until_node_id = "conversation_choice",
                run_num_trees = 1,
                reasoning = """
                Since we are looking at discussion within emails from Vin, I should return full_conversation, to give the user a full picture of the conversations.
                The user is looking for a summary of Vin's emails, so I should include the entire conversation.
                """,
                task = "full_conversation",
                all_actions_completed_reasoning = """
                The user will not be satisfied, because I have not yet replied to the user.
                After this task will be completed, I will have retrieved the emails, but I still need to write a response to them.
                Since there is another action I can perform, the user will not be satisfied.
                """,
                possible_tasks = ["full_conversation"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "Tell me about what Edward said in his most recent message", 
                run_until_node_id = "conversation_choice",
                run_num_trees = 1,
                reasoning = """
                Since we are looking at messages from Edward, I should return messages_only.
                The user is looking for a summary of only Edward's messages, so there is no need to include information 
                from messages that Edward has not written. In the future, I will need to respond to the user.
                """,
                task = "messages_only",
                all_actions_completed_reasoning = """
                The user will not be satisfied, because I have not yet responded to them, giving them a breakdown of the most recent message.
                The task will retrieve the information, which will be displayed to the user, but as a helpful assistant, I should respond to them.
                Since there is another action I can perform, the user will not be satisfied.
                """,
                possible_tasks = ["messages_only"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "summarize the issue related to 'PDFs being too large to upload'", 
                run_until_node_id = "base",
                run_num_trees = 1,
                reasoning = """
                Since we are looking at a GitHub issue, I should query the issue.
                Later, I will need to summarize the issue, but right now I should retrieve the information.
                """,
                task = "query",
                all_actions_completed_reasoning = """
                The user will not be satisfied, as the next task involves choosing the collection, so the information will not be retrieved until later.
                Additionally, I have not yet provided a summary of the issue.
                There are many more actions I can perform, so the user will not be satisfied.
                """,
                possible_tasks = ["query"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "Hi there, what can you do?", 
                run_until_node_id = "base",
                run_num_trees = 1,
                reasoning = """
                Since the user is asking about what the model can do, I should provide a text response in a chat interface.
                I should detail the capabilities of the Elysia app, and how it can help the user.
                """,
                task = "text_response",
                all_actions_completed_reasoning = """
                The user will be satisfied, as I will have provided a response to their question.
                There are no other actions I can perform, so this is the only option and hence the user will be satisfied.
                """,
                possible_tasks = ["text_response", "summarize"],
                all_actions_completed = True
            ),
            create_example(
                user_prompt = "What is the weather in Tokyo?", 
                run_until_node_id = "base",
                run_num_trees = 1,
                reasoning = """
                Since the user is asking about the weather in Tokyo, I should return a text response.
                I should provide a text response in a chat interface, and detail the weather in Tokyo, if I can.
                Unfortunately, I cannot provide the weather in Tokyo, so I will need to provide a text response.
                """,
                task = "text_response",
                all_actions_completed_reasoning = """
                The user will be satisfied, as I will respond to their question.
                There are no other actions I can perform, so this is the only option and hence the user will be satisfied.
                """,
                possible_tasks = ["text_response"],
                all_actions_completed = True
            ),
            create_example(
                user_prompt = "summarize the latest issues related to the verba app",
                run_until_node_id = "base",
                run_num_trees = 2,
                reasoning = """
                I have already retrieved the information about the latest issues related to the verba app, so I should summarize it.
                This is one of the actions I can perform, and is the last step to complete the overall goal.
                """,
                task = "summarize",
                all_actions_completed_reasoning = """
                The user will be satisfied, as I will have summarized the latest issues related to the verba app.
                There are no other actions I can perform, so this is the only option and hence the user will be satisfied.
                """,
                possible_tasks = ["summarize", "text_response"],
                all_actions_completed = True
            ),
            create_example(
                user_prompt = "Tell me about the latest scientific developments in AI", 
                run_until_node_id = "base",
                run_num_trees = 3,
                reasoning = """
                I have already attempted to retrieve information from the databases about this, but the available information is not relevant.
                Further, the collection names available in future actions are not relevant to the user's question.
                So there is nothing we can do to help the user with their question.
                Therefore, I will provide a text response to the user, and in it, I will state that I cannot help with their question.
                """,
                task = "text_response",
                all_actions_completed_reasoning = """
                The user will be satisfied, as there are no other actions I can perform.
                With this action, I will have provided a text response to the user, and in it, I state that I cannot help with their question.
                Therefore, the user will be satisfied.
                """,
                possible_tasks = ["text_response", "summarize"],
                all_actions_completed = True
            ),
            create_example(
                user_prompt = "How are you doing today?", 
                run_until_node_id = "base",
                run_num_trees = 1,
                reasoning = """
                Since the user is asking how I am doing, I should provide a text response.
                I should provide a text response in a chat interface, and detail how I am doing.
                """,
                task = "text_response",
                all_actions_completed_reasoning = """
                The user will be satisfied, as I will have provided a response to their question.
                There are no other actions I can perform, so this is the only option and hence the user will be satisfied.
                """,
                possible_tasks = ["text_response"],
                all_actions_completed = True
            ),
            create_example(
                user_prompt = "Give me every piece of information you have about the 'PDFs being too large to upload' issue in verba", 
                run_until_node_id = "collection",
                run_num_trees = 1,
                reasoning = """
                Since the user is asking for information about the 'PDFs being too large to upload' issue in verba, I should query any one of these collections.
                I will now choose the github issues collection, but in the future I should choose the slack conversations and email chains collection, 
                using the same query for these different collections.
                """,
                task = "example_verba_github_issues",
                all_actions_completed_reasoning = """
                The user will not be satisfied, as I have not yet provided the information to them.
                """,
                possible_tasks = ["example_verba_github_issues", "example_verba_slack_conversations", "example_verba_email_chains"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "Give me every piece of information you have about the 'PDFs being too large to upload' issue in verba", 
                run_until_node_id = "collection",
                run_num_trees = 2,
                reasoning = """
                The user is asking for information about the 'PDFs being too large to upload' issue in verba, I should query any one of these collections.
                I will now choose the email chains collection, but in the future I should choose the slack conversations or github issues collection, depending on which 
                one hasn't been queried yet. I will use the same query for these different collections.
                """,
                task = "example_verba_email_chains",
                all_actions_completed_reasoning = """
                The user will not be satisfied, as I have not yet provided the information to them in text format, but only the requested information.
                Later I will need to choose summarize to provide the information to the user in text format.
                """,
                possible_tasks = ["example_verba_github_issues", "example_verba_slack_conversations", "example_verba_email_chains"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "Give me every piece of information you have about the 'PDFs being too large to upload' issue in verba", 
                run_until_node_id = "collection",
                run_num_trees = 3,
                reasoning = """
                Looking at the data_queried field, I have already retrieved the information about the 'PDFs being too large to upload' issue for some collections.
                I should now choose a different collection to retrieve the information, because the user is asking for all information I have.
                This should be the last query required, so later when given the opportunity, I should choose summarize to convey the information to the user.
                """,
                task = "example_verba_slack_conversations",
                all_actions_completed_reasoning = """
                The user will not be satisfied, as I have not yet provided the information to them in text format.
                """,
                possible_tasks = ["example_verba_github_issues", "example_verba_slack_conversations", "example_verba_email_chains"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "Summarise 20 of the most recent emails and slack messages sent by Danny",
                run_until_node_id = "collection",
                run_num_trees = 1,
                reasoning = """
                Since the user is asking for a summary of the most recent emails and slack messages sent by Danny, 
                I should query the email chains collection, or the slack conversations collection, depending on which one hasn't been queried yet.
                """,
                task = "example_verba_email_chains",
                all_actions_completed_reasoning = """
                The user will not be satisfied, as I have not yet provided the information to them in text format.
                """,
                possible_tasks = ["example_verba_email_chains", "example_verba_slack_conversations"],
                all_actions_completed = False
            ),
            create_example(
                user_prompt = "What was the last email sent by Danny?",
                run_until_node_id = "base",
                run_num_trees = 1,
                reasoning = """
                We have retrieved the email conversation that Danny was involved in. I can see that they have said they
                made progress on the data analysis, so I should respond this to the user. Therefore, I should choose the text_response task.
                """,
                task = "text_response",
                all_actions_completed_reasoning = """
                The user will be satisfied, as I will have provided a response to their question.
                There are no other actions I can perform, so this is the last option and hence the decisions are completed.
                """,
                possible_tasks = ["text_response", "summarize"],
                all_actions_completed = True
            )
        ]
        
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), "w") as f:
            out = [example.toDict() for example in data]
            json.dump(out, f)

    else:
        with open(os.path.join(filepath, filename), "r") as f:
            data = [load_example_from_dict(example) for example in json.load(f)]

    return data


def train_decision_fewshot():
    
    train = make_examples()

    num_fewshot_demos = 12
    optimizer = LabeledFewShot(k=num_fewshot_demos)
    trained_fewshot = optimizer.compile(
        DecisionExecutor().activate_assertions(), 
        trainset=train
    )

    # Create the full directory path
    filepath = os.path.join(
        "elysia",
        "training",
        "dspy_models",
        "decision"
    )
    os.makedirs(filepath, exist_ok=True)  # This creates the directory if it doesn't exist

    # Save the file in the created directory
    full_filepath = os.path.join(filepath, f"fewshot_k{num_fewshot_demos}.json")
    trained_fewshot.save(full_filepath)

def train_decision_bootstrap_random_fewshot():
    train = make_examples()

    bootstrap = BootstrapFewShotWithRandomSearch(
        metric                 = metric, 
        max_labeled_demos      = 12, 
        max_bootstrapped_demos = 6, 
        num_candidate_programs = 6
    )
    
    model = DecisionExecutor().activate_assertions()

    # example1 = train[0].toDict()
    # del example1["reasoning"]
    # del example1["all_actions_completed_reasoning"]
    # del example1["all_actions_completed"]
    # del example1["task"]

    # model(**example1)

    # run optimiser
    bootstrap_trained_model = bootstrap.compile(
        student   = model, 
        trainset  = train
    )

    # Create the full directory path
    filepath = os.path.join(
        "elysia",
        "training",
        "dspy_models",
        "decision"
    )
    os.makedirs(filepath, exist_ok=True)  # This creates the directory if it doesn't exist

    # Save the file in the created directory
    full_filepath = os.path.join(filepath, f"bootstrap_random_fewshot.json")
    bootstrap_trained_model.save(full_filepath)


if __name__ == "__main__":
    
    # train_decision_fewshot()
    train_decision_bootstrap_random_fewshot()