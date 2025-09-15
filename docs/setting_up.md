
# Setting up Elysia

Elysia _requires_ setting up your LMs and API keys for the decision tree functionality to work. Additionally, to use Elysia to its full potential (adaptively searching and retrieving Weaviate data), it requires a _preprocessing_ step.

Elysia can be configured in three different ways: via the _configure_ function, by creating a _Settings_ object, or by setting _environment variables_ (in a `.env` file).

## Model Setup

Elysia uses two language models for different types of tasks;

* The **base model** is responsible for the decision agent, as well as any tools that specify its use.
* The **complex model** is used for more complex tasks, and is responsible for any tools that specify its use (such as the inbuilt query and aggregate tools).


### Configuring Models

To configure different LMs as default for all functions within Elysia, you can use the global [`configure` function](Reference/Settings.md#elysia.config.configure). For example, to use the GPT family of models, you can set:

```python
from elysia import configure

configure(
    base_model="gpt-4.1-mini",
    base_provider="openai",
    complex_model="gpt-4.1",
    complex_provider="openai",
    openai_api_key="..." # replace with your API key
)
```
The `configure` function can be used to specify both the `base_model` and `complex_model`. Both require separately setting a provider; in this case `openai` Instead, you can create your own `Settings` object which can be passed to any of the Elysia functions that use LMs to have a separate settings instance for each initialisation. E.g.,

```python
from elysia import Settings, Tree
my_settings = Settings()
tree = Tree(settings=my_settings)
```
This tree will use the `my_settings` object instead of the global one. If not specified, it will use the global settings. You can configure `my_settings` manually, either by `my_settings.configure(...)` (which takes exactly the same arguments as `configure`), or by using `my_settings.smart_setup()`, which uses recommended models based on the API keys and/or models set in the `.env` file, prioritising Gemini 2.0 Flash for the base and complex model. [See the reference page for more details.](Reference/Settings.md#elysia.config.Settings)

The third alternative: you can set everything in advance via creating a `.env` file in the root directory of your working directory, including the models, providers, and api keys. For example:

```
BASE_MODEL=gpt-4.1-mini
BASE_PROVIDER=openai
COMPLEX_MODEL=gpt-4.1
COMPLEX_PROVIDER=anthropic
OPENAI_API_KEY=... # replace with your OpenAI API key
```

Then, the global `settings` object will always use these values, and the `smart_setup()` or `my_settings.smart_setup()` (local settings object) will use these models and providers instead of the recommended ones.

### Local Model Integration via Ollama

First, make sure your Ollama server is running either via the Ollama app or `ollama run <model_name>`. E.g., `ollama run gpt-oss:20b`, which we'll use in this example. Within Python, you can configure your model API base to your Ollama api endpoint (default to `http://localhost:11434`) via the `model_api_base` parameter of `configure`.

```python 
from elysia import configure
configure(
    base_provider="ollama",
    complex_provider="ollama",
    base_model="gpt-oss:20b",
    complex_model="gpt-oss:20b",
    model_api_base="http://localhost:11434",
)
```

On the app side, this is configurable via the 'Api Base URL' parameter in the Settings. Set both of your providers to `ollama`, and your base and complex model to whatever model you are currently hosting, and this should work out-of-the-box.

Warning: Elysia uses a *long context*, quite long context, due to the nature of the collection schemas, environment and more being included in every prompt. So these models will run quite slowly. However, on the backend, you can configure this to be faster by disabling connection to your Weaviate cluster, if applicable, by removing your weaviate api key and url. Also, there is an optional setting
```python
settings.configure(
	base_use_reasoning=False,
	complex_use_reasoning=False
)
```
which will remove chain of thought prompting for the base and complex model, respectively. Use this with caution though, as it will degrade accuracy significantly. Additionally, some smaller models struggle with the complex nature of multiple outputs in DSPy and Elysia, so you might encounter some errors. In testing, the `gpt-oss` models work relatively well.

## Weaviate Integration

To use Elysia with Weaviate, you need to specify your Weaviate cluster details. These can be set via the Weaviate Cluster URL (`WCD_URL`) and the Weaviate Cluster API Key (`WCD_API_KEY`). To set these values, you can use `configure` on the settings:
```python
from elysia import configure
configure(
    wcd_url=..., # replace with your WCD_URL
    wcd_api_key=... # replace with your WCD_API_KEY
)
```
or by setting them as environment variables
```
WCD_URL=... # replace with your WCD_URL
WCD_API_KEY=... # replace with your WCD_API_KEY
```

Additionally, you need to _preprocess_ your collections for Elysia to use the built in Weaviate-based tools, see below for details.

*Note: using a local Weaviate instance is currently not supported. This is coming soon! [You can sign up for a 14-day sandbox for free.](https://weaviate.io/deployment/serverless)

## Preprocessing Collections

[The `preprocess` function](Reference/Preprocessor.md) must be used on the Weaviate collections you plan to use within Elysia. 

```python
from elysia import preprocess
preprocess("<your_collection_name>")
```

Preprocessing does several things:

- Creates an LLM generated summary of the collection, including descriptions of the fields in the dataset.
- Creates 'mappings', so that fields in the collection can be mapped to frontend-specific fields. This enables the Elysia frontend app to display items from the collection when retrieved in the app.
- Calculates summary statistics, such as the mean, maximum and minimum values of number fields, as well as statistics for other fields.
- Collects other metadata such as any named vectors, what index types are used, if the inverted index is configured to index e.g. creation time.

Since preprocessing uses LLM created summaries of the collections, you must configure your models in advance. [See above for details](#model-setup).

### Running the Preprocessing Function

You have access to two functions, `preprocess_async`, which must be awaited, and `preprocess`, which is a sync wrapper for its async sister. The basic arguments for either function are:

- **`collection_names`** (*list[str])*: The names of the collections to preprocess.
- **`client_manager`** *(ClientManager)*: The client manager to use.
    The ClientManager class is how Elysia interacts with Weaviate client.
    If you are unsure of this, do not provide this argument, it will default to the Weaviate cluster you selected via the `Settings`, or via `configure`/environment variables.

As well, the LLM requires a number of objects retrieved from the collection, at random, to help provide its summary. Since objects in collections vary greatly in token size (and hence LLM compute time/cost), you can adjust the following parameters to change how many objects are used for this sample.

- **`min_sample_size`** *(int)*: The minimum number of objects in the sample.
- **`max_sample_size`** *(int)*: The maximum number of objects to sample.
- **`num_sample_tokens`** *(int)*: The maximum number of tokens in the sample objects used to evaluate the summary.

The `num_sample_tokens` parameter controls how many objects are actually used. Provided it is between `min_sample_size` and `max_sample_size`, the preprocessor will select the closest number of objects that are estimated to be in total `num_sample_tokens` tokens.

Additionally, you have:
- **`settings`** *(Settings)*: The settings to use.
- **`force`** *(bool)*: Whether to force the preprocessor to run even if the collection already exists.

### Additional Functions

You can also use [`preprocessed_collection_exists`](Reference/Preprocessor.md#elysia.preprocessing.collection.preprocessed_collection_exists), which returns True/False if the collection has been preprocessed (and it can be accessed within the Weaviate cluster):

```python
from elysia import preprocessed_collection_exists
preprocessed_collection_exists(collection_name = ...)
```
which returns True/False if the preprocess exists within this Weaviate cluster

You can use [`edit_preprocessed_collection`](Reference/Preprocessor.md#elysia.preprocessing.collection.edit_preprocessed_collection) to update the values manually:
```python
from elysia import edit_preprocessed_collection
properties = edit_preprocessed_collection(
    collection_name = ...,
    named_vectors = ...,
    summary = ...,
    mappings = ...,
    fields = ...
)
```
which will change the LLM generated values with manually input values. Any fields not provided will not be updated.

You can use [`delete_preprocessed_collection`](Reference/Preprocessor.md#elysia.preprocessing.collection.delete_preprocessed_collection) which will delete the cached preprocessed metadata.

```python
delete_preprocessed_collection(collection_name = ...) 
```
which permanently deletes the preprocessed collection (not the original collection). You will need to rerun preprocess for the original collection to be used for the Weaviate integration in Elysia again.
