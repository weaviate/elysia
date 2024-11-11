
import os
import sys
import time
sys.path.append(os.getcwd())
os.chdir("../..")

from elysia.tree.tree import Tree
from elysia.tree.tree import lm

if __name__ == "__main__":

    tree = Tree(
        collection_names=["example_verba_github_issues", "example_verba_slack_conversations", "example_verba_email_chains"],
        verbosity=2, 
        break_down_instructions=False, 
        dspy_model=None
    )

    # tree.process(
    #     "List the most common issues from the verba github issues collection from 2024, sort by the most recent."
    # )

    # tree.returns.retrieved["example_verba_github_issues"].return_value(3)

    tree.process_sync(
        "What is Elysia?"
    )

    tree.returns