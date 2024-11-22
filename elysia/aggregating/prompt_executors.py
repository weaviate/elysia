import dspy

from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference
from elysia.util.logging import backend_print
from elysia.aggregating.prompt_templates import AggregatePrompt, construct_aggregate_initialiser_prompt

from weaviate.classes.query import Filter, Sort, Metrics
from weaviate.classes.aggregate import GroupByAggregate

class AggregateInitialiserExecutor(dspy.Module):
    
    def __init__(self, collection_names: list[str]):
        self.aggregate_initialiser_prompt = dspy.ChainOfThought(construct_aggregate_initialiser_prompt(collection_names))
        self.collection_names = collection_names

    def forward(self,
            user_prompt: str, 
            data_queried: list,
            current_message: str,
            previous_reasoning: dict
        ) -> str:
        
        return self.aggregate_initialiser_prompt(
            user_prompt=user_prompt, 
            reference=create_reference(),
            previous_reasoning=previous_reasoning,
            data_queried=data_queried,
            available_collections=self.collection_names,
            current_message=current_message,
        )

class AggregateExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.aggregate_prompt = dspy.ChainOfThought(AggregatePrompt)

    def _execute_code(self, aggregation_code: str, collection_name: str) -> dict:

        collection = client.collections.get(collection_name)

        if aggregation_code.startswith("```python") and aggregation_code.endswith("```"):
            aggregation_code = aggregation_code[8:-3]
        elif aggregation_code.startswith("```") and aggregation_code.endswith("```"):
            aggregation_code = aggregation_code[3:-3]
   
        return eval(aggregation_code)

    def forward(
        self, 
        user_prompt: str, 
        data_types: list, 
        example_field: dict, 
        previous_reasoning: dict, 
        previous_aggregations: list,
        collection_name: str
    ) -> str:
        
        prediction = self.aggregate_prompt(
            user_prompt=user_prompt, 
            reference=create_reference(), 
            previous_reasoning=previous_reasoning,
            data_types=data_types, 
            example_field=example_field, 
        )


        try:
            is_aggregation_possible = eval(prediction.is_aggregation_possible)
            assert isinstance(is_aggregation_possible, bool)
        except Exception as e:
            try:
                dspy.Assert(False, f"Error getting is_aggregation_possible: {e}", target_module=self.aggregate_prompt)
            except Exception as e:
                backend_print(f"Error getting is_aggregation_possible: {e}")
                # Return empty values when there's an error
                return None, None

        if not is_aggregation_possible:
            return None, None

        dspy.Suggest(
            prediction.code not in previous_aggregations,
            f"The aggregation code you have produced: {prediction.code} has already been used. Please produce a new aggregation code.",
            target_module=self.aggregate_prompt
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, collection_name)
        except Exception as e:

            try:
                # assert will raise an error if its failed multiple times
                dspy.Assert(False, f"Error executing aggregation code:\n{prediction.code}\nERROR: {e}", target_module=self.aggregate_prompt)
            except Exception as e:
                # in which case we just print the error and return 0 objects
                backend_print(f"Error executing aggregation code: {e}")
                return None, None
            
        return response, prediction