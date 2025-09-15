import dspy
from typing import List
from elysia.tools.text.objects import TextWithCitation


class CitedSummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects from the environment, summarize the information in the objects to answer the user's prompt.
    Your output will be a collection of TextWithCitation objects that form a complete summary when combined.
    Do not list any of the retrieved objects in your response. Do not give an itemised list of the objects, since they will be displayed to the user anyway.
    Your summary should be parsing the retrieved objects, and summarising the information in them that is relevant to the user's prompt.
    For example, picking the most relevant information from the retrieved objects, and discussing it, repeating ONLY relevant information if necessary.
    You should provide useful analysis, new information via analysing the existing objects, and synthesising the information.
    """

    subtitle = dspy.OutputField(description="A subtitle for the summary")
    cited_text: List[TextWithCitation] = dspy.OutputField(
        description="""
        A list of TextWithCitation objects whose 'text' fields will be concatenated to form the complete summary.
        
        FORMATTING RULES:
        1. Use markdown for structure (headings, lists, etc.), this is encouraged and recommended to break down the information to be more readable.
        2. IMPORTANT: Always add newline characters (\\n) before and after markdown elements (lists, headers)
        3. When a formatting element spans multiple TextWithCitation objects, ensure each object has proper newlines
        4. Example: If a bullet point list spans multiple objects, each object must end with \\n and the next begin with proper indentation
        
        CITATION RULES:
        1. NEVER include reference IDs in the 'text' field itself
        2. NEVER include the literal string "_REF_ID" in your text
        3. NEVER create hyperlinks to references in the text field
        4. ALWAYS copy reference IDs exactly from the source's _REF_ID to the ref_ids list
        5. Keep the text clean and reference-free: all citations happen only through the ref_ids list
        6. Since you are referencing via _REF_ID, you do not need any other type of referencing such as URLs or hyperlinks.

        CORRECT EXAMPLE: text: "Here is some text", ref_ids: ["example_citation_1", "example_citation_2"]
        INCORRECT EXAMPLE: text: "Here is some text _REF_ID", ref_ids: ["example_citation_1", "example_citation_2"]
        INCORRECT EXAMPLE: text: "Here is some text [example_citation_1]", ref_ids: ["example_citation_1", "example_citation_2"]
        
        CONTENT GUIDELINES:
        1. Focus only on information relevant to the user's prompt
        2. Provide analysis and synthesis, not just extraction
        3. Only cite information that exists in the retrieved objects
        4. Each text segment should map logically to its citations
        5. Use multiple TextWithCitation objects to align text with its specific sources
        6. Do not just repeat information from the environment. Create insights, offer suggestions, do not just list objects.
        """.strip()
    )


class SummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects, summarize the information in the objects to answer the user's prompt.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to summarize the information in the retrieved objects to answer the user's prompt.
    Do not list any of the retrieved objects in your response. Do not give an itemised list of the objects, since they will be displayed to the user anyway.
    """

    subtitle = dspy.OutputField(description="A subtitle for the summary")
    summary = dspy.OutputField(
        description="""
        The summary of the retrieved objects. You can use markdown formatting.
        Don't provide an itemised list of the objects, since they will be displayed to the user anyway.
        Your summary should take account what the user prompt is, and the information in the retrieved objects.
        DO NOT list the retrieved objects in your summary.
        Your summary should be parsing the retrieved objects, and summarising the information in them that is relevant to the user's prompt.
        For example, picking the most relevant information from the retrieved objects, and discussing it, repeating ONLY relevant information if necessary.
        """.strip()
    )


class TextResponsePrompt(dspy.Signature):
    """
    You are a helpful assistant, designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    Given a user_prompt, as well as a list of retrieved objects, respond to the user's prompt.
    Your response should be informal, polite, and assistant-like.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to respond to the user's query.
    Do not list any of the retrieved objects in your response. Do not give an itemised list of the objects, since they will be displayed to the user anyway.
    If there is an error and you could not complete a task, use this tool to suggest a brief reason why.
    If, for example, there is a missing API key, then the user needs to add it to the settings (which you should inform them of).
    Or you cannot connect to weaviate, then the user needs to input their API keys in the settings.
    If there are no collections available, the user needs to analyze this in the 'data' tab.
    If there are other problems, and it looks like the user can fix it, then provide a suggestion.
    """

    response = dspy.OutputField(
        description="""
        The response to the user's prompt.
        If you are explaining how something went wrong, or you could not complete a task, suggest a brief reason why.
        You can suggest alternative questions for the user to ask, alternative ways of phrasing the prompt, or explain your own limitations and how to avoid them.
        E.g., you could ask the user to be more specific, or to provide more information, or explain that you are limited by your data. Be creative and adaptive based on the user's prompt and the type of issue that occurred.
        Use present tense in your text, as if you are currently completing the action.
        Use gender neutral language.
        """.strip()
    )
