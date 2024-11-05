
import os
import sys
import time
sys.path.append(os.getcwd())
os.chdir("../..")

from backend.tree.tree import Tree
from backend.tree.tree import lm

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

    start = time.perf_counter()
    returns = tree.process(
        "Summarise some solutions people have proposed in slack for the issue of PDFs being too large to upload. don't forget to summarise!"
    )
    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")
    # print(returns)
