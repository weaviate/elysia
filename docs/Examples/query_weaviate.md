# Querying a Weaviate Database

This example will walk through using Elysia to:

* Setting up your API keys with Elysia for using models
* Setting up your Weaviate collections for usage with Elysia
* Query or aggregate your Weaviate collections
* How the decision tree works

## Before You Begin

Before setting up your environment and connecting to Weaviate, make sure you have the necessary API keys and access credentials for both your language models and your Weaviate instance. This will ensure a smooth setup process in the following steps.

1. [You should have a Weaviate cloud cluster - see Step 1.1 in the Weaviate quickstart guide.](https://docs.weaviate.io/weaviate/quickstart#step-1-set-up-weaviate)
2. [You need to find your REST endpoint URL and Admin API key for your cluster - see Step 1.3 in the Weaviate quickstart guide](https://docs.weaviate.io/weaviate/quickstart#13-connect-to-weaviate)
3. You additionally need API keys for any LLMs you want to use. We recommend [OpenRouter](https://openrouter.ai/) to gain access to a range of models.

## Setting up

Let's use the basic elysia `configure` to both *set up your models* and *connect to your Weaviate cluster*.

```python
from elysia import configure
configure(
    wcd_url = "...", # replace with your Weaviate REST endpoint URL
    wcd_api_key = "..." # replace with your Weaviate cluster API key,
    base_model = "gemini-2.0.flash-001",
    base_provider = "gemini",
    complex_model = "gemini-2.5.flash-001",
    complex_provider = "gemini",
    gemini_api_key = "..." # replace with your GEMINI_API_KEY from Google AI studio
)
```
Alternatively, you can use different models, such as `gpt-4.1-mini`, `gpt-4.1`, with `base_provider="openai"` and `complex_provider="openai"`, as well as an `openai_api_key`. Or any model/provider combination that you wish, [see the full LiteLLM docs for all API keys and models/providers](https://docs.litellm.ai/docs/providers).

## Optional: Add some data to your Weaviate cluster


We're going to create some basic data and an example collection for this demo. This is based on [this example in the Weaviate docs](https://docs.weaviate.io/weaviate/recipes/generative_search_aws_bedrock).

If you want to skip this step and use data from your own collection, simply replace all instances of the collection name "JeopardyQuestion" with your true collection name in later steps.

1. Download the example dataset.
    ```python
    import requests, json
    url = "https://raw.githubusercontent.com/weaviate/weaviate-examples/main/jeopardy_small_dataset/jeopardy_tiny.json"
    resp = requests.get(url)
    data = json.loads(resp.text)
    ```
    This dataset contains questions, answers and categories from Jeopardy questions. E.g.
    ```
    {
        "Category": "SCIENCE",
        "Question": "This organ removes excess glucose from the blood & stores it as glycogen",
        "Answer": "Liver"
    }
    ```

2. Import the data into Weaviate
    ```python
    from elysia.util.client import ClientManager

    client_manager = ClientManager()

    with client_manager.connect_to_client() as client:

        if client.collections.exists("JeopardyQuestion"):
            client.collections.delete("JeopardyQuestion")

        client.collections.create(
            "JeopardyQuestion"
        )

        jeopardy = client.collections.get("JeopardyQuestion")
        response = jeopardy.data.insert_many(data)

        if response.has_errors:
            print(response.errors)
        else:
            print("Insert complete.")
    ```
    This will by default use the settings inherited from the earlier `configure` function, so the Weaviate REST endpoint URL and API key set up previously.

## Preprocessing with Elysia

Now that you are fully set up with models and Weaviate integrations, you can move onto preprocessing your collection for use with Elysia. This is as simple as:
```python
from elysia import preprocess
preprocess("JeopardyQuestion")
```

<details closed>
<summary>View the Preprocessed Data</summary>

To view the preprocessing that has been completed, you can run the `view_preprocessed_collection` function on your collection:

```python
from elysia import view_preprocessed_collection
view_preprocessed_collection("JeopardyQuestion")
```
```json
{
    "mappings": {
        "document": {"content": "question", "category": "category", "title": "", "author": "", "date": ""},
        "table": {"category": "category", "question": "question", "answer": "answer"}
    },
    "prompts": [
        "What are some questions about DNA?",
        "What questions are in the SCIENCE category?",
        "What questions are in the ANIMALS category?",
        "What are some questions about mammals?",
        "What are some questions about snakes?",
        "What are the answers related to science?",
        "What are the answers related to animals?",
        "What questions involve the atmosphere?",
        "What questions involve metals?",
        "What questions involve organs?"
    ],
    "fields": [
        {
            "range": [1.0, 4.0],
            "type": "text",
            "groups": [
                {"count": 1, "value": "DNA"},
                {"count": 1, "value": "the atmosphere"},
                {"count": 1, "value": "wire"},
                {"count": 1, "value": "Elephant"},
                {"count": 1, "value": "Antelope"},
                {"count": 1, "value": "species"},
                {"count": 1, "value": "Liver"},
                {"count": 1, "value": "Sound barrier"},
                {"count": 1, "value": "the diamondback rattler"},
                {"count": 1, "value": "the nose or snout"}
            ],
            "mean": 1.7,
            "date_range": None,
            "name": "answer",
            "date_median": None,
            "description": "The correct response to the question posed in the 'question' field. This is a string 
containing the answer."
        },
        {
            "range": [1.0, 1.0],
            "type": "text",
            "groups": [{"count": 6, "value": "SCIENCE"}, {"count": 4, "value": "ANIMALS"}],
            "mean": 1.0,
            "date_range": None,
            "name": "category",
            "date_median": None,
            "description": "The subject area or topic to which the question and answer belong. Examples include 
'SCIENCE' and 'ANIMALS'."
        },
        {
            "range": [10.0, 22.0],
            "type": "text",
            "groups": [
                {
                    "count": 1,
                    "value": "A metal that is 'ductile' can be pulled into this while cold & under pressure"
                },
                {
                    "count": 1,
                    "value": "The gavial looks very much like a crocodile except for this bodily feature"
                },
                {
                    "count": 1,
                    "value": "In 1953 Watson & Crick built a model of the molecular structure of this, the 
gene-carrying substance"
                },
                {
                    "count": 1,
                    "value": "Weighing around a ton, the eland is the largest species of this animal in Africa"
                },
                {
                    "count": 1,
                    "value": "2000 news: the Gunnison sage grouse isn't just another northern sage grouse, but a 
new one of this classification"
                },
                {"count": 1, "value": "It's the only living mammal in the order Proboseidea"},
                {"count": 1, "value": "This organ removes excess glucose from the blood & stores it as glycogen"},
                {
                    "count": 1,
                    "value": "In 70-degree air, a plane traveling at about 1,130 feet per second breaks it"
                },
                {"count": 1, "value": "Heaviest of all poisonous snakes is this North American rattlesnake"},
                {"count": 1, "value": "Changes in the tropospheric layer of this are what gives us weather"}
            ],
            "mean": 15.0,
            "date_range": None,
            "name": "question",
            "date_median": None,
            "description": "The question or prompt for which the 'answer' field provides the correct response. This
is a string containing the question."
        }
    ],
    "summary": "This dataset contains questions and answers across various categories, primarily focusing on 
science and animals. Each entry includes a question, its corresponding answer, and the category to which the 
question belongs. The dataset provides a diverse set of trivia-like information suitable for quizzes or educational
purposes. The sample represents the entire dataset. The 'question' field is related to the 'answer' field, as the 
'answer' provides the correct response to the 'question'. The 'category' field classifies the 'question' and 
'answer' pair into a specific subject area. The category helps to group questions of similar topics together. The 
data is structured as a list of JSON objects. Each object contains three fields: 'answer', 'category', and 
'question'. No irregularities found. ",
    "vectorizer": None,
    "name": "JeopardyQuestion",
    "named_vectors": [
        {
            "source_properties": None,
            "enabled": True,
            "name": "default",
            "model": "Snowflake/snowflake-arctic-embed-l-v2.0",
            "description": "",
            "vectorizer": "TEXT2VEC_WEAVIATE"
        }
    ],
    "index_properties": {"isTimestampIndexed": False, "isNullIndexed": False, "isLengthIndexed": False},
    "length": 10.0
}
```
</details>


## Creating the Decision Tree

Now that the models and Weaviate integrations are set up (with `configure`), the default parameters to run the decision tree will also work automatically. 

Let's first create the class and inspect some of the properties.
```python
from elysia import Tree
tree = Tree()
```
<details closed>
<summary>Inspecting the Decision Tree Structure</summary>
To look at what tools are currently on the tree, we can inspect use the `tree.view()` method:

```python
print(tree.view())
```

```
📁 Base (base)
  ├── 🔧 Cited summarize (cited_summarize)
      💬 Summarize retrieved information for the user when all relevant data has
         been gathered. Provides a text response, and may end the conversation, but
         unlike text_response tool, can be used mid-conversation. Avoid for general
         questions where text_response is available. Summarisation text is directly
         displayed to the user. Most of the time, you can choose end_actions to be
         True to end the conversation with a summary. This is a good way to end the
         conversation.


  ├── 🔧 Text response (text_response)
      💬 End the conversation. This should be used when the user has finished their
         query, or you have nothing more to do except reply. You should use this to
         answer conversational questions not related to other tools. But do not use
         this as a source of information. All information should be from the
         environment if answering a complex question or an explanation. If there is
         an error and you could not complete a task, use this tool to suggest a
         brief reason why. If, for example, there is a missing API key, then the
         user needs to add it to the settings (which you should inform them of). Or
         you cannot connect to weaviate, then the user needs to input their API
         keys in the settings. If there are no collections available, the user
         needs to analyze this in the 'data' tab. If there are other problems, and
         it looks like the user can fix it, then provide a suggestion.


  ├── 🔧 Aggregate (aggregate)
      💬 Query the knowledge base specifically for aggregation queries. Performs
         calculations (counting, averaging, summing, etc.) and provides summary
         statistics on data. It can group data by properties and apply filters
         directly, without needing a prior query. Aggregation queries can be
         filtered. This can be applied directly on any collections in the schema.
         Use this tool when you need counts, sums, averages, or other summary
         statistics on properties in the collections. 'aggregate' should be
         considered the first choice for tasks involving counting, summing,
         averaging, or other statistical operations, even when filtering is
         required.


  ├── 🔧 Base.query (base.query)
      💬 Retrieves and displays specific data entries from the collections. Then,
         query with semantic search, keyword search, or a combination of both.
         Queries can be filtered, sorted, and more. Retrieving and displaying
         specific data entries rather than performing calculations or summaries. Do
         not use 'query' as a preliminary filtering step when 'aggregate' can
         achieve the same result more efficiently (if 'aggregate' is available).

    └── 🔧 Query postprocessing (query_postprocessing)
        💬 If the user has requested itemised summaries for retrieved objects, this
           tool summarises each object on an individual basis.


  └── 🔧 Visualise (visualise)
      💬 Visualise data in a chart from the environment. You can only visualise
         data that is in the environment. If there is nothing relevant in the
         environment, do not choose this tool.
```

These are the default tools available in a regular initialisation of the Elysia Tree, as well as their tool descriptions. To change the default tools available on a tree, you can initialise the tree with a different `branch_initialisation`, e.g.

```python
tree = Tree(branch_initialisation="empty")
```
will create a tree with no tools, and you can add custom tools via `tree.add_tool()`.

</details>


## Running the Decision Tree

To run the tool-running pipeline of the Elysia decision tree, you can simply call the class, i.e.

```python
response, objects = tree(
    "Find a single question about Science",
    collection_names = ["JeopardyQuestion"]
)
```

<details closed>
<summary>Real time updates</summary>
The default behaviour is that Elysia will print updates on what it is doing. In this example, this is
```
╭──────────── User prompt ─────────────╮
│                                      │
│ Find a single question about Science │
│                                      │
╰──────────────────────────────────────╯
╭───────────────────────────── Assistant response ─────────────────────────────╮
│                                                                              │
│ I will now search for a science question in the JeopardyQuestion collection. │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────── Current Decision ────────────────────────────────────────────────╮
│                                                                                                                 │
│ Node: base                                                                                                      │
│ Decision: query                                                                                                 │
│ Reasoning: The user is asking for a question about science.                                                     │
│ The `JeopardyQuestion` collection contains questions and answers, and the category field indicates whether the  │
│ question is about science.                                                                                      │
│ Therefore, I should query the `JeopardyQuestion` collection and filter for questions where the category is      │
│ science.                                                                                                        │
│ I should use the `query` tool to retrieve the questions.                                                        │
│                                                                                                                 │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────── Assistant response ───────────────────────────────────────────────╮
│                                                                                                                 │
│ I am now retrieving a single science question from the JeopardyQuestion collection by filtering for the         │
│ 'SCIENCE' category.                                                                                             │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────── JeopardyQuestion (Weaviate Query) ───────────╮
│                                                         │
│ collection.query.fetch_objects(                         │
│     filters=Filter.all_of([                             │
│         Filter.by_property('category').equal('SCIENCE') │
│     ]),                                                 │
│     limit=1                                             │
│ )                                                       │
│                                                         │
╰─────────────────────────────────────────────────────────╯
╭─────────────────────────────────────── Current Decision ────────────────────────────────────────╮
│                                                                                                 │
│ Node: base.query                                                                                │
│ Decision: query_postprocessing                                                                  │
│ Reasoning: Only one option available: query_postprocessing (and no function inputs are needed). │
│                                                                                                 │
│                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────── Current Decision ────────────────────────────────────────────────╮
│                                                                                                                 │
│ Node: base                                                                                                      │
│ Decision: text_response                                                                                         │
│ Reasoning: I have already retrieved a science question from the JeopardyQuestion collection in the previous     │
│ turn. The question is: "This organ removes excess glucose from the blood & stores it as glycogen". The answer   │
│ is "Liver". I should now respond to the user with this question.                                                │
│                                                                                                                 │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────── Assistant response ──────────────────────────────────────────────╮
│                                                                                                                │
│ Here's a science question for you: "This organ removes excess glucose from the blood & stores it as glycogen?" │
│                                                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
</details>

This will run through the decision tree, where the decision agent will choose tools based on what tools are available, what branch the decision agent is currently on, and context.


The `response` will be a concatenation of all responses that the model output (at each decision step and during any other response outputs):
```python
print(response)
```
```
'I will now search for a science question in the JeopardyQuestion collection. I am now retrieving a single science question from the JeopardyQuestion collection by filtering for the \'SCIENCE\' category. Here\'s a science question for you: "This organ removes excess glucose from the blood & stores it as glycogen?"'
```

The `objects` will be a list of objects that were added to the environment during that tree run:
```python
print(objects)
```
```
[
    [
        {
            'category': 'SCIENCE',
            'question': 'This organ removes excess glucose from the blood & stores it as glycogen',
            'answer': 'Liver',
            'uuid': 'b28ca48a-9a8d-417c-9ed1-e487132740ed',
            'collection_name': 'JeopardyQuestion',
            'chunk_spans': [],
            '_REF_ID': 'query_JeopardyQuestion_0_0'
        }
    ]
]
```

In this case, the model searched Weaviate using a `fetch_objects` search (no hybrid or semantic search), and used a `limit=1`, so only returned one object. The decision agent recognised that the query tool was called successfully, added a single object to the environment, and then finally recognised that the task was completed and therefore finalised the process with a `text_response`, informing the user of what the question was.

Note that the query tool was able to correctly use the filter `Filter.by_property('category').equal('SCIENCE')`, as the preprocessing step identified the unique groups for the property `'category'`, which was handed down to the query tool.

## Continuing the Conversation

By calling the same `Tree` class a second time, the conversation history is automatically included in the context of the decision tree. So if you were to ask:
```python
tree("What about animals?")
```

<details closed>
<summary>Real time updates</summary>
```
╭──── User prompt ────╮
│                     │
│ What about animals? │
│                     │
╰─────────────────────╯
╭──────────────────────────────── Assistant response ────────────────────────────────╮
│                                                                                    │
│ I will now search for a question about animals in the JeopardyQuestion collection. │
│                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────── Current Decision ────────────────────────────────────────────────╮
│                                                                                                                 │
│ Node: base                                                                                                      │
│ Decision: query                                                                                                 │
│ Reasoning: The user is now asking about animals, following a previous question about science.                   │
│ The `JeopardyQuestion` collection contains questions and answers, and the category field indicates whether the  │
│ question is about animals.                                                                                      │
│ Therefore, I should query the `JeopardyQuestion` collection and filter for questions where the category is      │
│ animals.                                                                                                        │
│ I should use the `query` tool to retrieve the questions.                                                        │
│                                                                                                                 │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────── Assistant response ───────────────────────────────────────────────╮
│                                                                                                                 │
│ I am now retrieving a single question about animals from the JeopardyQuestion collection by filtering for the   │
│ 'ANIMALS' category.                                                                                             │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────── JeopardyQuestion (Weaviate Query) ───────────╮
│                                                         │
│ collection.query.fetch_objects(                         │
│     filters=Filter.all_of([                             │
│         Filter.by_property('category').equal('ANIMALS') │
│     ]),                                                 │
│     limit=1                                             │
│ )                                                       │
│                                                         │
╰─────────────────────────────────────────────────────────╯
╭─────────────────────────────────────── Current Decision ────────────────────────────────────────╮
│                                                                                                 │
│ Node: base.query                                                                                │
│ Decision: query_postprocessing                                                                  │
│ Reasoning: Only one option available: query_postprocessing (and no function inputs are needed). │
│                                                                                                 │
│                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────── Current Decision ────────────────────────────────────────────────╮
│                                                                                                                 │
│ Node: base                                                                                                      │
│ Decision: text_response                                                                                         │
│ Reasoning: I have already retrieved a question about animals from the JeopardyQuestion collection in the        │
│ previous turn. The question is: "It's the only living mammal in the order Proboseidea". The answer is           │
│ "Elephant". I should now respond to the user with this question.                                                │
│                                                                                                                 │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── Assistant response ─────────────────────────────────────╮
│                                                                                             │
│ Here is an animal question for you: "It's the only living mammal in the order Proboseidea?" │
│                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────╯
```
</details>

Then the decision agent is aware that the context of the first question, i.e. "retrieve a single question", and uses that to frame its responses and queries for the second question, changing the filter from `"SCIENCE"` to `"ANIMALS"`. The conversation history can be viewed by running:

```python
print(tree.conversation_history)
```
```
[
    {'role': 'user', 'content': 'Find a single question about Science'},
    {
        'role': 'assistant',
        'content': 'I will now search for a science question in the JeopardyQuestion collection. I am now 
retrieving a single science question from the JeopardyQuestion collection by filtering for the \'SCIENCE\' 
category. Here\'s a science question for you: "This organ removes excess glucose from the blood & stores it as 
glycogen?"'
    },
    {'role': 'user', 'content': 'What about animals?'},
    {
        'role': 'assistant',
        'content': 'I will now search for a question about animals in the JeopardyQuestion collection. I am now 
retrieving a single question about animals from the JeopardyQuestion collection by filtering for the \'ANIMALS\' 
category. Here\'s a question about animals for you: "It\'s the only living mammal in the order Proboseidea?"'
    }
]
```