
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
        "Give me all the information about the issue related to PDFs being too large to upload in verba, as well as what people are saying about it."
    )
