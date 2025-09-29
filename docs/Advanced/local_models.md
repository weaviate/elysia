# Local Models 

Elysia integrates its LLM connections via [DSPy](https://dspy.ai/), which uses [LiteLLM](https://www.litellm.ai/) under the hood. The easiest way to get connected is via [Ollama](https://ollama.com/), but it is also possible to connect to OpenAI compatible endpoints. 

## Getting Connected

### Connecting via Ollama

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

### Connecting via OpenAI-Compatible Endpoints (Experimental)

See the [LiteLLM docs](https://docs.litellm.ai/docs/providers/openai_compatible) for more detail on using OpenAI compatible endpoints. In short, you can set your provider to `openai` and create a fake API key (e.g. `OPENAI_API_KEY=fake-key`) to enable connection to an OpenAI endpoint that is not hosted by OpenAI. You will also need to supply a `model_api_base` to the Elysia config which will point towards where your model is hosted. E.g.

```python
from elysia import configure
configure(
    base_provider="openai",
    complex_provider="openai",
    base_model="<your_model_name_here>",
    complex_model="<your_model_name_here>",
    openai_api_key="...",
    model_api_base="..."
)
```

## Warning

Elysia uses a *long context*, quite long context, due to the nature of the collection schemas, long instruction sets, environment and more being included in every prompt. So these models can and will probably run quite slowly, if hosted on a machine with low compute power (e.g. not a high powered GPU). 

## Recommendations

### Model Choice

The [`gpt-oss` family of models](https://ollama.com/library/gpt-oss) have been shown to work well with Elysia, and can handle the structured outputs well. These are trained specifically for agentic tasks and reasoning chains.

Please let us know on the [GitHub discussions](https://github.com/weaviate/elysia/discussions) if you've had any success with any other models.

### Speeding Up

**Disabling complex Weaviate integration to Elysia**

Within the python package, you can configure the Elysia process to be faster by disabling connection to your Weaviate cluster, if applicable, by removing your weaviate api key and url. Or, there is an optional setting on initialising the `Tree` to disable using Weaviate collections in Elysia:

```python
from elysia import Tree
tree = Tree(use_elysia_collections=False)
```

Setting this to `False` (default `True`) will disable the Elysia decision agent having access to the [preprocessed schemas](https://weaviate.github.io/elysia/setting_up/#preprocessing-collections) for any connected Weaviate collections. If you are not using a complex Weaviate integration, then this is safe to disable.

*Note that this also disables the inbuilt Query and Aggregate tools that Elysia is by default initialised to. If you are doing this, you should also add your own custom tools to Elysia for it to be worth anything! [See here for an intro to creating tools.](https://weaviate.github.io/elysia/creating_tools/)*

**Disabling chain of thought reasoning (Experimental)**

One of the biggest slowdowns of LMs is the number of output tokens they produce. There is an experimental configuration option for removing models from outputting chain of thought reasoning at every step, done so via:

```python
from elysia import configure
configure(
	base_use_reasoning=False,
	complex_use_reasoning=False
)
```

You can choose to disable just for the base model (e.g. the decision agent) or just the complex model (which some tools will use). Custom tools can also make use of the base/complex models also via the custom DSPy Module [`ElysiaChainOfThought`](https://weaviate.github.io/elysia/Reference/Util/#elysia.util.elysia_chain_of_thought.ElysiaChainOfThought).

*Use this with caution* - it will degrade accuracy significantly. 

**Future Plans**

Future versions of Elysia will hopefully include a simplified version of the system instructions and inputs to the decision agent/tools, that can be enabled via a flag in configure, e.g. `configure(..., simplify=True)` that will shrink the context size cleverly.

Stay tuned for more improvements coming to local models in Elysia by following/starring the [GitHub repository](https://github.com/weaviate/elysia). Or feel free to make a contribution!

## Troubleshooting

### When using local models, Elysia times out in the app, and I get an error

Try configuring your Tree Timeout in the configuration page to be higher. If a single request takes longer than this value, the conversation will time out and lead to an error.

### I'm getting random errors that don't seem to make sense

This could be one of many things:

- The conversation could be timing out (see above)
- Your model connection is failing
- A smaller local model may be failing to include every output in the response, or failing the structured output of DSPy. Try a larger model if you can with the same prompt, to see if the error persists.  If all the errors continue happening, [open a GitHub issue](https://github.com/weaviate/elysia/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen)!

### Nothing is helping, Elysia still isn't running with my local model

In Python, test the connection to your local model through [LiteLLM directly](https://docs.litellm.ai/docs/providers/ollama):

```python
from litellm import completion

response = completion(
    model="ollama/gemma3:4b", # or whichever model you are using
    messages=[{ "content": "hi", "role": "user"}], 
    api_base="http://localhost:11434"
)
print(response)
```

If the response is failing, then there is likely a problem with your connection to your model or Ollama (or very unlikely, LiteLLM). If this works, then try the connection in Elysia:

```python
from elysia import Tree, Settings

settings = Settings()
settings.configure(
    base_model="gemma3:4b",    # or whichever model you are using
    complex_model="gemma3:4b", # or whichever model you are using
    base_provider="ollama",
    complex_provider="ollama",
    model_api_base="http://localhost:11434", # or wherever your Ollama instance is 
)
tree = Tree(settings=settings)
```

Then:

```python
print(tree.base_lm("hi")) # should be a generic response without using elysia
```

This should be a direct calling of the LM, so a quick response with not a large amount of input tokens. Then you can run the decision tree:

```python
tree("hi") 
```

This now includes all context:

- System instructions for the decision tree
- Tool descriptions
- Conversation history
- Items in the internal environment (can be very large after processing requests like queries)
- Collection schemas (this is a big one)

So the request will take a lot longer. Leave this for as long as it needs. It might take a while - that's fine because we are just testing the connection.

If the model works (via `tree.base_lm("hi")`) but this step errors, it is either the model doing something wrong, or another error.

If it doesn't look like the model is doing something wrong, [open a GitHub issue](https://github.com/weaviate/elysia/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen), including a full error log from the python terminal.

If it is just taking a long time, then you may want to try a smaller model (not recommended currently) or finding access to some larger compute.

