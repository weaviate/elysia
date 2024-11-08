
import os
import sys
import time
sys.path.append(os.getcwd())
os.chdir("../..")

from elysia.tree.tree import Tree
from elysia.tree.tree import lm

if __name__ == "__main__":

    tree = Tree(verbosity=2, break_down_instructions=False, dspy_model="bootstrap_random_fewshot")



    # tree.process(
    #     "List the most common issues from the verba github issues collection from 2024, sort by the most recent."
    # )

    # tree.returns.retrieved["example_verba_github_issues"].return_value(3)

    tree.process_sync(
        "What was the last email sent by Danny?"
    )

    tree.returns