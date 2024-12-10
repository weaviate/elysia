import dspy
import datetime
import re
import time
from typing import Any, Generator

# Weaviate functions for code execution
from weaviate.classes.query import Filter, Sort, QueryReference
from weaviate.collections.classes.internal import QueryReturn
from weaviate.classes.aggregate import GroupByAggregate

# Chunking
from elysia.preprocess.chunk import CollectionChunker

# Globals
from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference

# Prompt Templates
from elysia.querying.prompt_templates import (
    construct_query_prompt, 
    ObjectSummaryPrompt 
)

# dspy
from elysia.dspy.environment_of_thought import EnvironmentOfThought

# Objects
from elysia.api.objects import Warning, Error, Status, TreeUpdate

# Util
from elysia.util.logging import backend_print
from elysia.util.parsing import format_datetime

class SafetyException(Exception):
    def __init__(self, message: str):
        self.message = message

class QueryExecutor(dspy.Module):

    def __init__(self, collection_names: list[str] = None):
        super().__init__()
        self.query_prompt = EnvironmentOfThought(construct_query_prompt(collection_names))
        self.collection_names = collection_names

    def _evaluate_code_safety(self, query_code: str) -> tuple[bool, str]:
        # List of dangerous operations/keywords that should not be allowed
        dangerous_operations = [
            "delete", "drop", "remove", "update", "insert", "create", "batch",
            "configure", "schema", "exec", "eval", "import", "__", "os.", "sys.",
            "subprocess", "open", "write", "file", "globals", "locals", "getattr",
            "setattr", "delattr", "compile", "builtins", "breakpoint", "callable",
            "classmethod", "staticmethod", "super",
            "lambda", "async", "await", "yield", "with", "raise", "try", "except",
            "finally", "class", "def", "print", 
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

        # Outer brackets (that wrap the main function call)
        param_start = query_code.find("(")
        param_end = query_code.rfind(")")

        # If there are no outer brackets, there are no parameters
        if param_start == -1 or param_end == -1:
            return False, "No parameters detected."
        
        params_text = query_code[param_start+1:param_end]


        param_names = params_text.split("=")
        # param_names = [p.split(",") for p in param_names]
        # param_names = [item.strip() for sublist in param_names for item in sublist]
        # param_names = [p.split("\n") for p in param_names]
        # param_names = [item.strip() for sublist in param_names for item in sublist]
        # param_names = [p for p in param_names if p != "" and not p.startswith("#")]
        # param_names = [param_names[i].strip() for i in range(0, len(param_names), 2)]

        # params_list = [p.strip() for p in params_list if p.strip() != ""]
        # params_list = [p.split("\n") for p in params_list]
        # params_list = [item.strip() for sublist in params_list for item in sublist]
        # params_list = [p for p in params_list if "(" not in p and ")" not in p]

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

        for param in param_names:
            if param not in allowed_params:
                return False, f"Invalid parameter detected: {param}."

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

    def _update_limit(self, code, new_limit):
        return re.sub(r'limit=\d+', f'limit={new_limit}', code)

    def _extract_limit(self, code):
        match = re.search(r'limit=(\d+)', code)
        if match:
            return match.group(1)
        return None
    
    def _add_argument(self, code_string: str, arg_name: str, arg_value: str):
        # Find the last argument before the closing parenthesis
        code_string = code_string[:code_string.rfind(")")]
        
        # Replace with existing argument + comma + new argument
        replacement = f"{code_string},\n    {arg_name}={arg_value})"
        
        return replacement    

    def _execute_large_code(self, query_code: str, collection_name: str):
        # modify query code to get more objects for chunking
        current_limit = self._extract_limit(query_code)
        if current_limit is not None:
            modified_query = self._update_limit(query_code, min(int(current_limit)*5, 100)) # TODO: make this dynamic
            modified_query = self._add_argument(
                modified_query, 
                "return_references", 
                f'QueryReference(link_on="isChunked", return_properties=[])'
            )

            return self._execute_code(modified_query, collection_name)
        else:
            return self._execute_code(query_code, collection_name)

    def _get_mapping(self, collection_name: str, collection_information: dict):
        return collection_information[collection_name]["mappings"]

    def _evaluate_content_field(self, return_type: str, mappings: dict) -> str:
        """
        Return the original object field if its mapped to 'content', otherwise return an empty string.
        If no content field mapping exists in `mappings`, it will return an empty string anyway (by default).
        """

        if return_type == "conversation":
            return mappings[return_type]["content"]
        elif return_type == "message":
            return mappings[return_type]["content"]
        elif return_type == "ticket":
            return mappings[return_type]["content"]
        elif return_type == "ecommerce":
            return mappings[return_type]["description"]
        elif return_type == "epic_generic":
            return mappings[return_type]["content"]
        elif return_type == "document":
            return mappings[return_type]["content"]
        else:
            return ""
    
    def _find_query_type(self, query_code: str) -> str:

        query_code = query_code.strip()
        start_pos = query_code.find("collection.query.") + len("collection.query.")
        end_pos = query_code.find("(", start_pos)
        
        if start_pos == -1 or end_pos == -1:
            return "fetch_objects"  # default fallback
            
        # Extract the query type
        query_type = query_code[start_pos:end_pos].strip()
        
        return query_type

    def _get_chunked_collection_name(self, collection_name: str):
        return f"ELYSIA_CHUNKED_{collection_name}__"
    
    def _evaluate_needs_chunking(
            self, 
            content_field: str,
            collection_information: dict, 
            collection_name: str,
            query_code: str,
            threshold: int = 200
        ) -> bool:
        return (
            content_field != "" and 
            collection_information[collection_name]["fields"][content_field]["mean"] > threshold and
            self._find_query_type(query_code) != "fetch_objects"
        )
    
    def set_collection_names(self, collection_names: list[str]):
        self.collection_names = collection_names
        self.query_prompt = EnvironmentOfThought(construct_query_prompt(collection_names))

    async def forward(
        self, 
        user_prompt: str, 
        previous_queries: list, 
        conversation_history: list[dict],
        data_queried: list,
        previous_reasoning: dict,
        collection_information: list,
        collection_return_types: dict[str, list[str]],
        current_message: str
    ):
        
        # run query code generation
        try:
            t = time.time()
            prediction = self.query_prompt(
                user_prompt=user_prompt, 
                reference=create_reference(),
                conversation_history=conversation_history,
                previous_reasoning=previous_reasoning,
                collection_information={
                    collection_name: {
                            k: v for k, v in collection_information.items() if k != "mappings"
                    } for collection_name, collection_information in collection_information.items()
                },
                previous_queries=previous_queries,
                current_message=current_message,
                collection_return_types=collection_return_types,
                data_queried=data_queried
            )
            print(f"Time taken for query creator prompt: {time.time() - t:.2f} seconds")
        except Exception as e:
            backend_print(f"Error in query creator prompt: {e}")
            # Return empty values when there's an error
            yield QueryReturn(objects=[])
            yield Error(f"Error in LLM call: {e}")
            return 

        is_query_possible = prediction.is_query_possible

        if not is_query_possible:
            yield False
            yield QueryReturn(objects=[])
            yield Warning("LLM deemed query as not possible, continuing.")
            return 

        dspy.Suggest(
            prediction.code not in previous_queries,
            f"The query code you have produced: {prediction.code} has already been used. Please produce a new query code.",
            target_module=self.query_prompt
        )

        dspy.Assert(
            prediction.return_type in collection_return_types[prediction.collection_name],
            f"""
            The return type you have produced: {prediction.return_type} is not one of the return types for the collection {prediction.collection_name}. 
            If you are choosing {prediction.collection_name}, then the return type must be one of {collection_return_types[prediction.collection_name]} exactly as it appears.
            Please produce a new return type.
            """.strip(),
            target_module=self.query_prompt
        )

        # once the collection name is determined, get the mapping to find the content field
        mapping = self._get_mapping(prediction.collection_name, collection_information)
        content_field = self._evaluate_content_field(prediction.return_type, mapping)

        # empty string = no mapping and can't chunk, so if it exists, let's evaluate whether it needs chunking
        # or if the query is fetch_objects, then we don't need to chunk, because we're doing SQL style query
        needs_chunking = self._evaluate_needs_chunking(
            content_field, 
            collection_information, 
            prediction.collection_name, 
            prediction.code
        )

        if needs_chunking and prediction.return_type == "document":
            print(f"Chunking {prediction.collection_name}")
            collection_chunker = CollectionChunker(prediction.collection_name)
            collection_chunker.create_chunked_reference(content_field)

            # TODO: add error catching here
            objects = self._execute_large_code(prediction.code, prediction.collection_name)
            
            try:
                objects = self._execute_large_code(prediction.code, prediction.collection_name)
            
            except SafetyException as e:
                try:
                    dspy.Assert(False, 
                                f"Safety exception: {e} for query code:\n{prediction.code}. The code was deemed unsafe.", 
                                target_module=self._execute_code
                                )
                
                except SafetyException as e:
                    backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
                    yield QueryReturn(objects=[])
                    yield Error(f"Safety error while executing code: {e}")
                
                except Exception as e:
                    backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
                    yield QueryReturn(objects=[])
                    yield Error(f"Error while executing code: {e}")        
            except Exception as e:

                try:
                    dspy.Assert(False, 
                                f"""
                                Error executing query code:
                                {prediction.code}
                                The ERROR message was: {e}
                                Ensure that the query code is valid python output.
                                Or you should set is_query_possible to False if you want to stop the query from being executed.
                                """
                                , 
                                target_module=self.query_prompt
                                )
                except SafetyException as e:
                    backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
                    yield QueryReturn(objects=[])
                    yield Error(f"Safety error while executing code: {e}")
                
                except Exception as e:
                    backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
                    yield QueryReturn(objects=[])
                    yield Error(f"Error while executing code: {e}")

            
            # yield TreeUpdate(
            #     from_node="query_executor",
            #     to_node="document_chunker",
            #     reasoning=f"Chunking {len(objects.objects)} objects from {prediction.collection_name}"
            # )
            yield Status(f"Chunking {len(objects.objects)} objects from {prediction.collection_name}")

            collection_chunker(objects, content_field); # this will insert the chunked objects into the chunked collection
            collection_to_query = self._get_chunked_collection_name(prediction.collection_name) # once objects are chunked, we query the chunked collection
            yield Status(f"Querying chunked {prediction.collection_name}")
        else:
            collection_to_query = prediction.collection_name
            # yield TreeUpdate(
            #     from_node="query_executor", 
            #     to_node="document_chunker", 
            #     reasoning="This step was skipped because it was determined that the text was not long enough to be chunked, or is not the right data format."
            # )
            yield Status(f"Querying collection {prediction.collection_name}")

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, collection_to_query)
        except SafetyException as e:
            try:
                dspy.Assert(False, 
                            f"Safety exception: {e} for query code:\n{prediction.code}. The code was deemed unsafe.", 
                            target_module=self._execute_code
                            )
            
            except SafetyException as e:
                backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
                yield QueryReturn(objects=[])
                yield Error(f"Safety error while executing code: {e}")
            
            except Exception as e:
                backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
                yield QueryReturn(objects=[])
                yield Error(f"Error while executing code: {e}")        
        except Exception as e:

            try:
                dspy.Assert(False, 
                            f"""
                            Error executing query code:
                            {prediction.code}
                            The ERROR message was: {e}
                            Ensure that the query code is valid python output.
                            Or you should set is_query_possible to False if you want to stop the query from being executed.
                            """
                            , 
                            target_module=self.query_prompt
                            )
            except SafetyException as e:
                backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
                yield QueryReturn(objects=[])
                yield Error(f"Safety error while executing code: {e}")
            
            except Exception as e:
                backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
                yield QueryReturn(objects=[])
                yield Error(f"Error while executing code: {e}")
            
        yield response
        yield prediction

    # def forward(
    #     self, 
    #     user_prompt: str, 
    #     previous_queries: list, 
    #     conversation_history: list[dict],
    #     data_queried: list,
    #     previous_reasoning: dict,
    #     collection_information: list,
    #     collection_return_types: dict[str, list[str]],
    #     current_message: str
    # ) -> tuple[QueryReturn, dict, str]:

    #     # run query code generation
    #     try:
    #         prediction = self.query_prompt(
    #             user_prompt=user_prompt, 
    #             reference=create_reference(),
    #             conversation_history=conversation_history,
    #             previous_reasoning=previous_reasoning,
    #             collection_information={
    #                 collection_name: {
    #                         k: v for k, v in collection_information.items() if k != "mappings"
    #                 } for collection_name, collection_information in collection_information.items()
    #             },
    #             previous_queries=previous_queries,
    #             current_message=current_message,
    #             collection_return_types=collection_return_types,
    #             data_queried=data_queried
    #         )
    #     except Exception as e:
    #         backend_print(f"Error in query creator prompt: {e}")
    #         # Return empty values when there's an error
    #         return QueryReturn(objects=[]), None, f"Error in LLM call: {e}"

    #     is_query_possible = prediction.is_query_possible

    #     if not is_query_possible:
    #         return QueryReturn(objects=[]), None, ""

    #     dspy.Suggest(
    #         prediction.code not in previous_queries,
    #         f"The query code you have produced: {prediction.code} has already been used. Please produce a new query code.",
    #         target_module=self.query_prompt
    #     )

    #     dspy.Assert(
    #         prediction.return_type in collection_return_types[prediction.collection_name],
    #         f"""
    #         The return type you have produced: {prediction.return_type} is not one of the return types for the collection {prediction.collection_name}. 
    #         If you are choosing {prediction.collection_name}, then the return type must be one of {collection_return_types[prediction.collection_name]} exactly as it appears.
    #         Please produce a new return type.
    #         """.strip(),
    #         target_module=self.query_prompt
    #     )

    #     # once the collection name is determined, get the mapping to find the content field
    #     mapping = self._get_mapping(prediction.collection_name, collection_information)
    #     content_field = self._evaluate_content_field(prediction.return_type, mapping)

    #     # empty string = no mapping and can't chunk, so if it exists, let's evaluate whether it needs chunking
    #     # or if the query is fetch_objects, then we don't need to chunk, because we're doing SQL style query
    #     needs_chunking = self._evaluate_needs_chunking(
    #         content_field, 
    #         collection_information, 
    #         prediction.collection_name, 
    #         prediction.code
    #     )

    #     if needs_chunking:
    #         # TODO: add error catching here
    #         objects = self._execute_large_code(prediction.code, prediction.collection_name)
    #         collection_chunker = CollectionChunker(prediction.collection_name)
    #         collection_chunker(objects, content_field); # this will insert the chunked objects into the chunked collection
    #         collection_to_query = self._get_chunked_collection_name(prediction.collection_name) # once objects are chunked, we query the chunked collection
    #     else:
    #         collection_to_query = prediction.collection_name


    #     # catch any errors in query execution for dspy assert
    #     try:
    #         response = self._execute_code(prediction.code, collection_to_query)
    #     except SafetyException as e:
    #         try:
    #             dspy.Assert(False, 
    #                         f"Safety exception: {e} for query code:\n{prediction.code}. The code was deemed unsafe.", 
    #                         target_module=self._execute_code
    #                         )
            
    #         except SafetyException as e:
    #             backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
    #             return QueryReturn(objects=[]), None, f"Safety error while executing code: {e}"
            
    #         except Exception as e:
    #             backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
    #             return QueryReturn(objects=[]), None, f"Error while executing code: {e}"        
    #     except Exception as e:

    #         try:
    #             dspy.Assert(False, 
    #                         f"""
    #                         Error executing query code:
    #                         {prediction.code}
    #                         The ERROR message was: {e}
    #                         Ensure that the query code is valid python output.
    #                         Or you should set is_query_possible to False if you want to stop the query from being executed.
    #                         """
    #                         , 
    #                         target_module=self.query_prompt
    #                         )
    #         except SafetyException as e:
    #             backend_print(f"[bold red]Safety error while executing code: {e}[/bold red]")
    #             return QueryReturn(objects=[]), None, f"Safety error while executing code: {e}"
            
    #         except Exception as e:
    #             backend_print(f"[bold red]Error while executing code: {e}[/bold red]")
    #             return QueryReturn(objects=[]), None, f"Error while executing code: {e}"
            
    #     return response, prediction, ""

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
    
