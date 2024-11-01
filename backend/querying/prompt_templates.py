import dspy

class QueryRewritingPrompt(dspy.Signature):
    """
    Given a sentence summary of a social media post, create a single search term to find relevant technical documents that will be used in writing the post.
    You are also given existing contexts that have been retrieved, and existing queries that have been generated.
    You should make your new query distinct from the existing queries, and in general should be looking for a new concept, either from the contexts (if you need additional information to understand the contexts themselves), or from the summary (if the post summary itself requires additional information).
    You should assume NO prior knowledge of the content in the summary or the contexts, so anything technical in these should be included in the query. Find the most relevant concept to the summary that is not yet included in the contexts to search for.
    Primarily, the query should be based on keywords in the summary/contexts.
    """
    user_prompt = dspy.InputField(desc="The user's original query")
    previous_queries = dspy.InputField(
        desc="""
        A comma separated list of existing queries that have been generated for the post. 
        This can be used to avoid generating duplicate queries. 
        If this field is an empty list, you are generating the first query.
        """.strip()
    )
    query = dspy.OutputField(
        desc="""
        a single item in quotes, i.e. a small sentence or phrase. 
        This will be used for semantic search on relevant content to include in the post. 
        This should be for _one_ concept only, do not make this query too complex, and consist of mostly keywords. 
        """.strip()
    )
