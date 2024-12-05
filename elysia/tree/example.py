
import os
import sys
import time
sys.path.append(os.getcwd())
os.chdir("../..")

from rich import print
from elysia.tree.tree import Tree
from elysia.tree.tree import fallback_lm

if __name__ == "__main__":

    tree = Tree(
        collection_names=[
            "example_verba_github_issues", 
            "example_verba_slack_conversations", 
            "example_verba_email_chains", 
            "ecommerce",
            "financial_contracts",
            "weather"
        ],
        verbosity=2, 
        break_down_instructions=False, 
        dspy_model=None
    )

    outputs = tree.process_sync(
        "what was kaladin's most recent message?",
        training_route="search/query/text_response",
    )

    print(tree.tree_data.data_queried_string())
    tree = Tree(
        collection_names=[
            "example_verba_github_issues", 
            "example_verba_slack_conversations", 
            "example_verba_email_chains", 
            "ecommerce",
            "financial_contracts",
            "weather"
        ],
        verbosity=2, 
        break_down_instructions=False, 
        dspy_model=None
    )

    outputs = tree.process_sync(
        "what was kaladin's most recent message?",
        training_route="text_response",
    )

    tree.base_lm.inspect_history(5)
    tree.complex_lm.inspect_history(5)

    print(tree.tree_data.data_queried_string())
    
    # print(tree.returns.retrieved["example_verba_github_issues"].objects)
    # print(tree.returns.aggregation["example_verba_github_issues"].metadata["last_code"])
    
    # tree.process_sync(
    #     "query again to find out who else was in the conversation about that that message was in?"
    # )

    # print(tree.returns)

    # tree.process_sync(
    #     "what is my name"
    # )
    
    # # tree.returns.retrieved["example_verba_slack_conversations"].objects

    # tree.process_sync(
    #     "can't you see the name in your previous reasoning history?"
    # )