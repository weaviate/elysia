
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
        "hi my name is danny"
    )

    print(tree.returns)

    tree.process_sync(
        "what is my name"
    )
    
    # tree.returns.retrieved["example_verba_email_chains"].metadata

    tree.process_sync(
        "can't you see the name in your previous reasoning history?"
    )