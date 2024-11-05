
import os
import sys
sys.path.append(os.getcwd())
os.chdir("../..")

from backend.tree.tree import Tree
import dspy
import weaviate
from weaviate.classes.init import Auth

from rich import print

# lm=dspy.LM(model="gpt-4o-mini", max_tokens=8000)
# dspy.settings.configure(lm=lm)


if __name__ == "__main__":

    # tree = Tree(verbosity=2)

    # returns = tree.process(
    #     "I want to know about the issues with the verba app, then summarise the information"
    # )
    # print(returns)

    # returns = tree.process(
    #     "Just list the issues related to 'PDFs being too large to upload'"
    # )
    # print(returns)


    tree = Tree(verbosity=1)

    returns = tree.process(
        "Summarise some solutions people have proposed in slack for the issue of PDFs being too large to upload"
    )
    # print(returns)