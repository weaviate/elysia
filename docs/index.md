# Welcome to Elysia

Elysia is an agentic platform designed to use tools in a decision tree. A decision agent decides which tools to use dynamically based on its environment and context. You can use custom tools or use the pre-built tools designed to retrieve your data in a Weaviate cluster.

See the [basic example to get started right away!](basic.md) Or if you want to make your own tools and customise Elysia, see how to [easily add your own tools](creating_tools.md).

## Get Started

To use Elysia, you need to either set up your models and API keys in your `.env` file, or specify them in the config. [See the setup page to get started.](setting_up.md)

Elysia can be used very simply:
```python
from elysia import tool, Tree

tree = Tree()

@tool(tree=tree)
async def add(x: int, y: int) -> int:
    return x + y

tree("What is the sum of 9009 and 6006?")
```

Elysia is pre-configured to be capable of connecting to and interacting with your [Weaviate](https://weaviate.io/deployment/serverless) clusters!
```python
from elysia import Tree
tree = Tree()
response, objects = tree(
    "What are the 10 most expensive items in the Ecommerce collection?",
    collection_names = ["Ecommerce"]
)
```
This will use the built-in open source _query_ tool or _aggregate_ tool to interact with your Weaviate collections. To get started connecting to Weaviate, [see the setting up page](setting_up.md#weaviate-integration).

## Installation

```bash
pip install elysia-ai
```

## Usage

Elysia is free, open source, and available to anyone.

Unlike other agent-based packages, Elysia is pre-configured to run a wide range of tools and has a lot of capabilities straight away. For example, you could just call Elysia on your Weaviate collections and it will immediately and dynamically search your data, using custom queries with filters or aggregations.

Or you could customise Elysia to your liking, create your own custom tools and add them to the Elysia decision tree.

To use Elysia to search your data, you need a Weaviate cluster (or you can define your own custom tool to search another data source!).

[Sign up to Weaviate! A 14 day sandbox cluster is free.](https://weaviate.io/deployment/serverless)

For more information on signing up to Weaviate, [click here](https://weaviate.io/developers/wcs/platform/create-account). 

From your weaviate cluster, you can upload data via a CSV on the cloud console, or [you can upload via the Weaviate APIs](https://weaviate.io/developers/academy/py/zero_to_mvp/schema_and_imports/import).

## About

Check out the Github Repositories for the backend and the frontend

- [elysia](https://github.com/weaviate/elysia) (backend)

- [elysia-frontend](https://github.com/weaviate/elysia-frontend) (frontend)

Elysia was developed by Edward Schmuhl (frontend) and Danny Williams (backend). Check out our socials below:


- [Edward's Linkedin](https://www.linkedin.com/in/edwardschmuhl/)

- [Danny's Linkedin](https://www.linkedin.com/in/dannyjameswilliams/)

Documentation built with [mkdocs](https://www.mkdocs.org).