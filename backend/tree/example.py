
import os
import sys
import time
sys.path.append(os.getcwd())
os.chdir("../..")

from backend.tree.tree import Tree
from backend.tree.tree import lm

if __name__ == "__main__":

    tree = Tree(verbosity=2, break_down_instructions=False)



    # tree.process(
    #     "List the most common issues from the verba github issues collection from 2024, sort by the most recent."
    # )

    # tree.returns.retrieved["example_verba_github_issues"].return_value(3)

    tree.process(
        "Create a summary of the last 10 messages sent by kaladin."
    )

    tree.returns.retrieved["example_verba_email_chains"].return_value(0)