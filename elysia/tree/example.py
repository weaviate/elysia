
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

    tree.process_sync(
        "hi"
    )

    # x=1
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