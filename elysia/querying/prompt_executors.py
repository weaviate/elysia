import dspy
import datetime
from typing import Any, Generator

# Weaviate functions for code execution
from weaviate.classes.query import Filter, Sort
from weaviate.collections.classes.internal import QueryReturn
from weaviate.classes.aggregate import GroupByAggregate

# Globals
from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference

# Prompt Templates
from elysia.querying.prompt_templates import (
    construct_query_prompt, 
    ObjectSummaryPrompt,  
)

# Util
from elysia.util.logging import backend_print
from elysia.util.parsing import format_datetime

class SafetyException(Exception):
    def __init__(self, message: str):
        self.message = message

class QueryExecutor(dspy.Module):

    def __init__(self, collection_names: list[str] = None, return_types: list[str] = None):
        super().__init__()
        self.query_prompt = dspy.ChainOfThought(construct_query_prompt(collection_names, return_types))
        self.available_collections = collection_names
        self.available_return_types = return_types

    def _evaluate_code_safety(self, query_code: str) -> tuple[bool, str]:
        # List of dangerous operations/keywords that should not be allowed
        dangerous_operations = [
            "delete", "drop", "remove", "update", "insert", "create", "batch",
            "configure", "schema", "exec", "eval", "import", "__", "os.", "sys.",
            "subprocess", "open", "write", "file", "globals", "locals", "getattr",
            "setattr", "delattr", "compile", "builtins", "breakpoint", "callable",
            "classmethod", "staticmethod", "super",
            "lambda", "async", "await", "yield", "with", "raise", "try", "except",
            "finally", "class", "def", "return", "print"
        ]

        # Maximum allowed query length
        MAX_QUERY_LENGTH = 1000
        if len(query_code) > MAX_QUERY_LENGTH:
            return False, f"Query is too long ({len(query_code)} > {MAX_QUERY_LENGTH})."

        # Check for multiple statements (allow semicolons within quotes)
        in_single_quote = False
        in_double_quote = False
        for i, char in enumerate(query_code):
            if char == "'" and (i == 0 or query_code[i-1] != "\\"):
                in_single_quote = not in_single_quote
            elif char == '"' and (i == 0 or query_code[i-1] != "\\"):
                in_double_quote = not in_double_quote
            elif char == ";" and not (in_single_quote or in_double_quote):
                return False, "Multiple statements detected (semi-colon outside of quotes)."

        # Convert to lowercase for case-insensitive checking
        query_code_lower = query_code.lower()
        
        # Check for dangerous operations outside of quotes
        for op in dangerous_operations:

            # Find all occurrences of the dangerous operation
            pos = 0
            while True:
                pos = query_code_lower.find(op, pos)
                if pos == -1:  # No more occurrences found
                    break
                    
                # Check if this occurrence is within quotes
                in_quotes = False
                in_single_quote = False
                in_double_quote = False
                
                for i in range(pos):
                    if query_code[i] == "'" and (i == 0 or query_code[i-1] != "\\"):
                        in_single_quote = not in_single_quote
                    elif query_code[i] == '"' and (i == 0 or query_code[i-1] != "\\"):
                        in_double_quote = not in_double_quote
                
                in_quotes = in_single_quote or in_double_quote
                
                if not in_quotes:
                    return False, f"Dangerous operation '{op}' detected (outside of quotes)."
                
                pos += 1  # Move to next position to continue search

        # Only allow specific query patterns
        allowed_patterns = [
            "collection.query"
        ]
        
        # Ensure the query starts with one of the allowed patterns
        if not any(query_code.strip().startswith(pattern) for pattern in allowed_patterns):
            return False, "Query does not start with one of the allowed patterns."
        
        # Ensure only allowed parameter names are used
        allowed_params = [
            "near_vector", "certainty", "distance", "limit", "offset", "auto_limit",
            "query", "filters", "limit", "offset", "sort", "group_by", "include", 
            "where", "near_vector", "near_object", "vector", "uuid", "rerank",
            "target_vector", "include_vector", "return_metadata", "return_properties",
            "return_references"
        ]
        
        # Extract parameter names from the query
        param_start = query_code.find("(")
        param_end = query_code.rfind(")")
        if param_start == -1 or param_end == -1:
            return False, "No parameters detected."
        
        params_text = query_code[param_start+1:param_end]
        param_names = [p.split("=")[0].strip() for p in params_text.split(",") if "=" in p]
        
        # Track bracket depth to only parse top-level parameters
        bracket_depth = 0
        current_param = ""
        param_names = []
        in_single_quote = False
        in_double_quote = False
        
        for char in params_text:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not (in_single_quote or in_double_quote):
                if char == '(':
                    bracket_depth += 1
                elif char == ')':
                    bracket_depth -= 1
                elif char == ',' and bracket_depth == 0:
                    if '=' in current_param and bracket_depth == 0:
                        param_names.append(current_param.split('=')[0].strip())
                    current_param = ""
                    continue
            
            current_param += char
            
        # Don't forget the last parameter
        if '=' in current_param and bracket_depth == 0:
            param_names.append(current_param.split('=')[0].strip())
            
        if not all(param in allowed_params for param in param_names):
            return False, "Invalid parameter detected."

        # Check parentheses balance
        if query_code.count("(") != query_code.count(")"):
            return False, "Parentheses are not balanced."
            
        return True, "Query is safe."   

    def _execute_code(self, query_code: str, collection_name: str) -> dict:

        collection = client.collections.get(collection_name)

        if query_code.startswith("```python") and query_code.endswith("```"):
            query_code = query_code[8:-3]
        elif query_code.startswith("```") and query_code.endswith("```"):
            query_code = query_code[3:-3]

        is_safe, reason = self._evaluate_code_safety(query_code)
        if not is_safe:
            raise SafetyException(f"Dangerous code detected. Halting execution.\nREASON: {reason}")
   
        return eval(query_code)

    def forward(
        self, 
        user_prompt: str, 
        previous_queries: list, 
        conversation_history: list[dict],
        data_queried: list,
        previous_reasoning: dict,
        collection_information: list,
        current_message: str
    ) -> Generator[Any, Any, Any]:

        # run query code generation
        try:
            prediction = self.query_prompt(
                user_prompt=user_prompt, 
                reference=create_reference(),
                conversation_history=conversation_history,
                previous_reasoning=previous_reasoning,
                collection_information=collection_information,
                previous_queries=previous_queries,
                current_message=current_message,
                data_queried=data_queried
            )
        except Exception as e:
            backend_print(f"Error in query creator prompt: {e}")
            # Return empty values when there's an error
            return QueryReturn(objects=[]), None, f"Error in LLM call: {e}"

        try:
            is_query_possible = eval(prediction.is_query_possible)
            assert isinstance(is_query_possible, bool)
        except Exception as e:
            try:
                dspy.Assert(False, f"Error getting is_query_possible: {e}", target_module=self.query_prompt)
            except Exception as e:
                backend_print(f"Error getting is_query_possible: {e}")
                return QueryReturn(objects=[]), None, f"Error in LLM call: {e}"

        if not is_query_possible:
            return QueryReturn(objects=[]), None, ""

        dspy.Suggest(
            prediction.code not in previous_queries,
            f"The query code you have produced: {prediction.code} has already been used. Please produce a new query code.",
            target_module=self.query_prompt
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, prediction.collection_name)
        except SafetyException as e:
            try:
                dspy.Assert(False, 
                            f"Safety exception: {e} for query code:\n{prediction.code}. The code was deemed unsafe.", 
                            target_module=self._execute_code
                            )
            
            except SafetyException as e:
                backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
                return QueryReturn(objects=[]), None, f"Safety error while executing code: {e}"
            
            except Exception as e:
                backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
                return QueryReturn(objects=[]), None, f"Error while executing code: {e}"
            
        except Exception as e:

            try:
                dspy.Assert(False, 
                            f"Error executing query code:\n{prediction.code}\nERROR: {e}", 
                            target_module=self.query_prompt
                            )
            except SafetyException as e:
                backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
                return QueryReturn(objects=[]), None, f"Safety error while executing code: {e}"
            
            except Exception as e:
                backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
                return QueryReturn(objects=[]), None, f"Error while executing code: {e}"
            
        return response, prediction, ""

class ObjectSummaryExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.object_summary_prompt = dspy.ChainOfThought(ObjectSummaryPrompt)

    def forward(self, objects: list[dict], current_message: str):
        prediction = self.object_summary_prompt(objects=objects, current_message=current_message)

        try:
            summary_list = eval(prediction.summaries)
        except Exception as e:
            dspy.Assert(False, f"Error converting summaries to list: {e}", target_module=self.object_summary_prompt)

        return summary_list, prediction