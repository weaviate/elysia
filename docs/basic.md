# Basic Example

Let's assume we have access to the following Weaviate collections:

- `ecommerce`: a fashion dataset with fields such as `price`, `category`, `description`, etc.

- `ml_wikipedia`: a collection of long Wikipedia articles related to machine learning, with fields such as `categories`, `content`, `title` etc.

- `Tickets`: Github issues for a fictional company

These Weaviate collections are those which we want to search over using Elysia.

## Setup

You need to specify what models you want to use, as well as any API keys. To set up the models, you can use `configure`. For example, if you want to use the GPT-4o family of models:

```python
from elysia import configure
configure(
    base_model="gpt-4o-mini",
    base_provider="openai",
    complex_model="gpt-4o",
    complex_provider="openai",
    openai_api_key="sk-...", # replace with your API key
    wcd_url="...",    # replace with your weaviate cloud url
    wcd_api_key="..." # replace with your weaviate cloud api key
)
```

You need to specify both a `base_model` and a `complex_model`, as well as their providers. This hooks into LiteLLM through DSPy, [so any LiteLLM supported models and providers will work](https://docs.litellm.ai/docs/providers). [See the setup page for more details](setting_up.md).

Then for a collection to be accessible within Elysia, we need to preprocess it - so that the models are aware of the schemas and information about the collection.

```python
from elysia import preprocess
preprocess("Tickets")
```
 
## Running Elysia

To run the Elysia decision tree, using the default setup, just call the `Tree` object!

```python
from elysia import Tree
tree = Tree()
response, objects = tree("what were the 10 most recent Github issues?")
```
Elysia will _dynamically_ run through the decision tree, choosing tools to use based on the decision agent LM based on the input. The decision tree returns the concatenation of all text responses from the models, as well as any retrieved objects (anything that was added to the environment during this call).

```python
print(response)
```
```
I will now query the Tickets collection to retrieve the 10 most recent issues for you. I applied a descending sort on the "issue_created_at" field to retrieve the 10 most recent issues. I will now summarize the 10 most recent Github issues for you. The latest tickets reflect ongoing discussions and developments within the Verba project. Notable entries include a closed issue regarding the use of specific model inputs, a report on a breaking change affecting the code chunker functionality, and requests for enhancements like custom JSON support and improved metadata handling during file uploads. The issues also highlight user concerns about application performance when processing large document uploads and the integration of external language models. Overall, these issues illustrate a dynamic environment with active contributions and feedback from the community.
```
```python
print(objects)
```
```
[
    [
        {
            'issue_id': 2843638219.0,
            'issue_content': "If you set OLLAMA_MODEL and OLLAMA_EMBED_MODEL they will be the ones suggested instead of the first on Ollama's list.",
            'issue_created_at': '2025-02-10T21:06:00Z',
            'issue_labels': [],
            'issue_url': 'https://github.com/weaviate/Verba/pull/372',
            'issue_comments': 0.0,
            'issue_title': 'use OLLAMA_MODEL OLLAMA_EMBED_MODEL as input suggestion when using Ollama',
            'issue_author': 'dudanogueira',
            'issue_updated_at': '2025-02-27T10:38:02Z',
            'issue_state': 'closed',
            'uuid': 'bc56b4b2fc6a541c94969721bd895a7c',
            'ELYSIA_SUMMARY': ''
        },
    ...
        {
            'issue_id': 2845625123.0,
            'issue_content': "Pull request for feature #2.",
            'issue_created_at': '2025-01-4T22:16:05Z',
            'issue_labels': [],
            'issue_url': 'https://github.com/weaviate/Verba/pull/373',
            'issue_comments': 2.0,
            'issue_title': 'Feature PR #2',
            'issue_author': 'thomashacker',
            'issue_updated_at': '2025-01-15T11:06:08Z',
            'issue_state': 'closed',
            'uuid': '05dae4214e9050a59d4e9985892cdc10',
            'ELYSIA_SUMMARY': ''
        }
    ]
]
```