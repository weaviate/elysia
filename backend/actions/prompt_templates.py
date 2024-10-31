import dspy

class QueryRewritingPrompt(dspy.Signature):
    """
    Given some user instructions, create a query hybrid search (semantic, vector space as well as keyword search) that will be used to retrieve the most relevant documents from a knowledge base.
    """
    user_prompt = dspy.InputField(description="The user's original query")
    query = dspy.OutputField(description="The rewritten query")
