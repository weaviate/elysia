import dspy

from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference
from elysia.util.logging import backend_print
from elysia.aggregating.prompt_templates import AggregatePrompt


class AggregateCollectionExecutor(dspy.Module):

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
        data_fields: list, 
        example_field: dict, 
        previous_reasoning: dict, 
        collection_name: str
    ) -> str:
        
        prediction = self.aggregate_prompt(
            user_prompt=user_prompt, 
            reference=create_reference(), 
            previous_reasoning=previous_reasoning,
            data_fields=data_fields, 
            example_field=example_field, 
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, collection_name)
        except Exception as e:

            try:
                # assert will raise an error if its failed multiple times
                dspy.Assert(False, f"Error executing aggregation code:\n{prediction.code}\nERROR: {e}", target_module=self.aggregate_collection_prompt)
            except Exception as e:
                # in which case we just print the error and return 0 objects
                backend_print(f"Error executing aggregation code: {e}")
                return None, None, None
            
        return response, prediction
