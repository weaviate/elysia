import dspy

# Globals
from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference

# dspy
from elysia.dspy.environment_of_thought import EnvironmentOfThought

# Util
from elysia.util.logging import backend_print

# Prompt Templates
from elysia.aggregating.prompt_templates import construct_aggregate_prompt

# Weaviate functions for code execution
from weaviate.classes.query import Filter, Sort, Metrics
from weaviate.classes.aggregate import GroupByAggregate

class AggregateExecutor(dspy.Module):

    def __init__(self, collection_names: list[str]):
        self.aggregate_prompt = EnvironmentOfThought(construct_aggregate_prompt(collection_names))
        self.collection_names = collection_names

    def _execute_code(self, aggregation_code: str, collection_name: str) -> dict:

        collection = client.collections.get(collection_name)

        if aggregation_code.startswith("```python") and aggregation_code.endswith("```"):
            aggregation_code = aggregation_code[8:-3]
        elif aggregation_code.startswith("```") and aggregation_code.endswith("```"):
            aggregation_code = aggregation_code[3:-3]
   
        return eval(aggregation_code)

    def set_collection_names(self, collection_names: list[str]):
        self.collection_names = collection_names
        self.aggregate_prompt = EnvironmentOfThought(construct_aggregate_prompt(collection_names))

    def forward(
        self, 
        user_prompt: str, 
        data_queried: list, 
        collection_information: list,
        previous_reasoning: dict, 
        previous_aggregations: list,
        conversation_history: list[dict]
    ) -> str:
        
        prediction = self.aggregate_prompt(
            user_prompt=user_prompt, 
            reference=create_reference(), 
            conversation_history=conversation_history,
            previous_reasoning=previous_reasoning,
            data_queried=data_queried,
            collection_information={
                collection_name: {
                        k: v for k, v in collection_information.items() if k != "mappings"
                } for collection_name, collection_information in collection_information.items()
            },
            previous_aggregations=previous_aggregations
        )

        if not prediction.is_aggregation_possible:
            return None, None, ""

        dspy.Suggest(
            prediction.code not in previous_aggregations,
            f"The aggregation code you have produced: {prediction.code} has already been used. Please produce a new aggregation code.",
            target_module=self.aggregate_prompt
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, prediction.collection_name)
        except Exception as e:

            try:
                # assert will raise an error if its failed multiple times
                dspy.Assert(False, 
                            f"""
                            Error executing aggregation code:
                            {prediction.code}
                            The error was: {e}
                            Ensure that the output of the code is correct, and that it is a valid Python query.
                            Do not assume any other functions except what you have been given and do not use any comments.
                            """.strip(), 
                            target_module=self.aggregate_prompt
                        )
            except Exception as e:
                # in which case we just print the error and return 0 objects
                backend_print(f"Error executing aggregation code: {e}")
                return None, None, f"Error executing aggregation code: {e}"
            
        return response, prediction, ""
