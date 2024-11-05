
import os
import sys
import time
sys.path.append(os.getcwd())
os.chdir("../..")

from backend.tree.tree import Tree
from backend.tree.tree import lm

if __name__ == "__main__":

    tree = Tree(verbosity=2, break_down_instructions=False)



    tree.process(
        "List 5 random issues from the verba github issues collection."
    )
