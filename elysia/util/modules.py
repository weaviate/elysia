import json
from typing import Type, Callable
from copy import copy, deepcopy
from typing import AsyncGenerator, Literal

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature
from dspy.streaming import StreamListener, StreamResponse

from elysia.tree.objects import TreeData, Atlas, Environment
from elysia.util.feedback import retrieve_feedback
from elysia.util.client import ClientManager
from elysia.util.parsing import estimate_tokens

elysia_meta_prompt = """
You are part of an ensemble of agents that are working together to solve a task.
Your task is just one as part of a larger system of agents designed to work together.
You will be given meta information about where you are in the larger system, 
such as what other tasks have been completed, any items or objects that have been collected during the full process.

Your job is not to answer the entire task, but to complete your part of the task.
Therefore the `user_prompt` should not be judged in its entirety, analyse what you can complete based on the prompt.
Complete this task only, remember that other agents will complete other parts of the task.
"""


class AssertionError(Exception):
    pass


class AssertedModule(dspy.Module):

    def __init__(
        self,
        module_or_signature: dspy.Module | type[dspy.Signature],
        assertion_function: Callable[[dspy.Prediction, dict], tuple[bool, str]],
        num_tries: int = 3,
    ):
        if isinstance(module_or_signature, dspy.Module):
            self.module = module_or_signature
        else:
            self.module = dspy.Predict(module_or_signature)
        self.assertion_function = assertion_function
        self.asserted_module = self._assert_module(self.module)
        self.num_tries = num_tries

    def _assert_module(self, module: dspy.Module) -> dspy.Module:
        FeedbackModule = deepcopy(module)

        if hasattr(module, "signature"):
            signature = FeedbackModule.signature
        elif hasattr(module, "predict"):
            signature = FeedbackModule.predict.signature

        signature = signature.prepend(  # type: ignore
            name="feedback",
            field=dspy.InputField(
                description=(
                    "Feedback from the previous attempt. "
                    "Reasoning why the previous outputs were incorrect. "
                    "Take this feedback into account and re-attempt the previous task."
                ),
            ),
        )

        signature = signature.prepend(  # type: ignore
            name="history",
            field=dspy.InputField(description=""),
            type_=dspy.History,
        )

        return FeedbackModule

    def forward(self, **kwargs) -> dspy.Prediction:
        prediction: dspy.Prediction = self.module.forward(**kwargs)  # type: ignore
        success, feedback = self.assertion_function(prediction, kwargs)
        history = dspy.History(messages=[])
        if not success:
            history.messages.append({**kwargs, **prediction})
            for attempt in range(self.num_tries):
                prediction: dspy.Prediction = self.asserted_module.forward(feedback=feedback, history=history, lm=kwargs["lm"])  # type: ignore
                success, feedback = self.assertion_function(prediction, kwargs)
                if success:
                    break
                else:
                    history.messages.append(
                        {**kwargs, "feedback": feedback, **prediction}
                    )
            if not success:
                raise AssertionError(feedback)
        return prediction

    async def aforward(self, **kwargs) -> dspy.Prediction:
        prediction: dspy.Prediction = await self.module.aforward(**kwargs)  # type: ignore
        success, feedback = self.assertion_function(prediction, kwargs)
        history = dspy.History(messages=[])
        if not success:
            history.messages.append({**kwargs, **prediction})
            for attempt in range(self.num_tries):
                prediction: dspy.Prediction = await self.asserted_module.aforward(feedback=feedback, history=history, lm=kwargs["lm"])  # type: ignore
                success, feedback = self.assertion_function(prediction, kwargs)
                if success:
                    break
                else:
                    history.messages.append(
                        {**kwargs, "feedback": feedback, **prediction}
                    )
            if not success:
                raise AssertionError(feedback)
        return prediction

    async def aforward_streaming(
        self, streamed_field: str, **kwargs
    ) -> AsyncGenerator[dspy.Prediction | StreamResponse, None]:

        if not hasattr(self.module, "aforward_streaming"):
            raise ValueError("Module has no attribute aforward_streaming!")

        found_pred = False
        async for chunk in self.module.aforward_streaming(streamed_field=streamed_field, **kwargs):  # type: ignore
            if isinstance(chunk, StreamResponse):
                yield chunk
            elif isinstance(chunk, dspy.Prediction):
                prediction = chunk
                found_pred = True

        if not found_pred:
            raise AssertionError("No prediction found during streaming.")

        success, feedback = self.assertion_function(prediction, kwargs)
        history = dspy.History(messages=[])

        if not success:
            history.messages.append({**kwargs, **prediction})
            for attempt in range(self.num_tries):
                found_pred = False
                async for chunk in self.asserted_module.aforward_streaming(streamed_field=streamed_field, **kwargs):  # type: ignore
                    if isinstance(chunk, StreamResponse):
                        yield chunk
                    elif isinstance(chunk, dspy.Prediction):
                        prediction = chunk
                        found_pred = True

                if not found_pred:
                    raise AssertionError("No prediction found during streaming.")

                success, feedback = self.assertion_function(prediction, kwargs)
                if success:
                    break
                else:
                    history.messages.append(
                        {**kwargs, "feedback": feedback, **prediction}
                    )

        if not success:
            raise AssertionError(feedback)

        yield prediction

    async def aforward_with_feedback_examples(
        self,
        **kwargs,
    ) -> tuple[dspy.Prediction, list[str]]:

        if not hasattr(self.module, "aforward_with_feedback_examples"):
            raise ValueError(
                "Module has no attribute aforward_with_feedback_examples, which is required if using "
                "the experimental feature USE_FEEDBACK. This is likely an issue with the Elysia versioning/install. "
                "Disable USE_FEEDBACK to prevent this error. "
            )

        prediction, uuids = await self.module.aforward_with_feedback_examples(**kwargs)  # type: ignore
        success, feedback = self.assertion_function(prediction, kwargs)
        history = dspy.History(messages=[])
        if not success:
            history.messages.append({**kwargs, **prediction})
            for attempt in range(self.num_tries):
                (
                    prediction,
                    uuids,
                ) = await self.asserted_module.aforward_with_feedback_examples(
                    feedback=feedback, history=history, lm=kwargs["lm"]
                )  # type: ignore
                success, feedback = self.assertion_function(prediction, kwargs)
                if success:
                    break
                else:
                    history.messages.append(
                        {**kwargs, "feedback": feedback, **prediction}
                    )
            if not success:
                raise AssertionError(feedback)

        return prediction, uuids

    async def aforward_streaming_with_feedback_examples(
        self, streamed_field: str, **kwargs
    ) -> AsyncGenerator[dspy.Prediction | StreamResponse | list[str], None]:

        if not hasattr(self.module, "aforward_streaming_with_feedback_examples"):
            raise ValueError(
                "Module has no attribute aforward_streaming_with_feedback_examples, which is required if using "
                "the experimental feature USE_FEEDBACK. This is likely an issue with the Elysia versioning/install. "
                "Disable USE_FEEDBACK to prevent this error. "
            )

        found_pred = False
        async for chunk in self.module.aforward_streaming_with_feedback_examples(
            streamed_field=streamed_field, **kwargs
        ):  # type: ignore
            if isinstance(chunk, StreamResponse):
                yield chunk
            elif isinstance(chunk, dspy.Prediction):
                prediction = chunk
                found_pred = True
            elif isinstance(chunk, list):
                uuids = chunk

        if not found_pred:
            raise AssertionError("No prediction found during streaming.")

        success, feedback = self.assertion_function(prediction, kwargs)
        history = dspy.History(messages=[])

        if not success:
            history.messages.append({**kwargs, **prediction})
            for attempt in range(self.num_tries):
                found_pred = False
                async for chunk in self.asserted_module.aforward_streaming(streamed_field=streamed_field, **kwargs):  # type: ignore
                    if isinstance(chunk, StreamResponse):
                        yield chunk
                    elif isinstance(chunk, dspy.Prediction):
                        prediction = chunk
                        found_pred = True
                    elif isinstance(chunk, list):
                        uuids = chunk

                if not found_pred:
                    raise AssertionError("No prediction found during streaming.")

                success, feedback = self.assertion_function(prediction, kwargs)
                if success:
                    break
                else:
                    history.messages.append(
                        {**kwargs, "feedback": feedback, **prediction}
                    )

        if not success:
            raise AssertionError(feedback)

        yield uuids
        yield prediction


class ElysiaPrompt(Module):
    """
    A custom reasoning DSPy module that reasons step by step in order to predict the output of a task.
    It will automatically include the most relevant inputs:
    - The user's prompt
    - The conversation history
    - The atlas
    - Any errors (from calls of the same tool)

    And you can also include optional inputs (by setting their boolean flags on initialisation to `True`):
    - The environment
    - The collection schemas
    - The tasks completed

    You can also specify `collection_names` to only include certain collections in the collection schemas.

    It will optionally output (by setting the boolean flags on initialisation to `True`):
    - The reasoning (model step by step reasoning)
    - A message update (if `message_update` is `True`), a brief 'update' message to the user.
    - Whether the task is impossible (boolean)

    You can use this module by calling the `.forward()` or `.aforward()` method, passing all your *new* inputs as keyword arguments.
    You do not need to include keyword arguments for the other inputs, like the `environment`.

    Example:

    ```python
    my_module = ElysiaPrompt(
        signature=...,
        tree_data=...,
        message_update=True,
        environment=True,
        collection_schemas=True,
        tasks_completed=True,
    )
    my_module.aforward(input1=..., input2=..., lm=...)
    ```
    """

    def __init__(
        self,
        signature: Type[Signature],
        tree_data: TreeData,
        reasoning: bool = True,
        impossible: bool = True,
        message_update: bool = True,
        environment_level: Literal["full", "metadata", "dynamic", "none"] = "dynamic",
        collection_schemas: bool = False,
        tasks_completed: bool = False,
        collection_names: list[str] = [],
        **config,
    ):
        """
        Args:
            signature (Type[dspy.Signature]): The signature of the module.
            tree_data (TreeData): Required. The tree data from the Elysia decision tree.
                Used to input the current state of the tree into the prompt.
                If you are using this module as part of a tool, the `tree_data` is an input to the tool call.
            reasoning (bool): Whether to include a reasoning input (chain of thought). Recommended for improved accuracy.
            impossible (bool): Whether to include a boolean flag indicating whether the task is impossible.
                This is useful for stopping the tree from continuing and returning to the base of the decision tree.
                For example, the model judges a query impossible to execute, or the user has not provided enough information.
            message_update (bool): Whether to include a message update input.
                If True, the LLM output will include a brief 'update' message to the user.
                This describes the current action the LLM is performing.
                Designed to increase interactivity and provide the user with information before the final output.
            environment_level (Literal["full", "metadata", "dynamic", "none"]): Whether to include an environment as an input.
                - `"full"` means the environment in its entirety will always be included in the prompt. Not recommended as this can hit context limits and cause errors.
                - `"metadata"` means the environment metadata only is displayed. This includes metadatas from the results themselves and how many objects of each type are in the environment.
                - `"dynamic"` means the environment metadata is displayed when a token limit is reached, and the full environment is shown otherwise. Recommended.
                  Additionally, if the decision agent has used a 'view_environment' tool to inspect the most relevant parts of the environment, these viewed objects will be passed down to this LM call instead.
                  The token limit is controlled by the `env_token_limit` parameter of the `Settings` class, configurable via the `.configure(env_token_limit=...)` method of `Settings`.
                - `"none"` means the environment is never included in the prompt
            collection_schemas (bool): Whether to include a collection schema input.
                If True, the module will include the preprocessed collection schemas in the prompt input.
                This is useful so that the LLM knows the structure of the collections, if querying or similar.
                Use this sparingly, as it will use a large amount of tokens.
                You can specify `collection_names` to only include certain collections in this schema.
            tasks_completed (bool): Whether to include a tasks completed input.
                If True, the module will include the list of tasks completed input.
                This is a nicely formatted list of the tasks that have been completed, with the reasoning for each task.
                This is used so that the LLM has a 'stream of consciousness' of what has already been done,
                as well as to stop it from repeating actions.
                Other information is included in the `tasks_completed` field that format outputs from previous tasks.
                This is useful for continuing a decision logic across tasks, or to reinforce key information.
            collection_names (list[str]): A list of collection names to include in the prompt.
                If provided, this will modify the collection schema input to only include the collections in this list.
                This is useful if you only want to include certain collections in the prompt.
                And to reduce token usage.
            **config (Any): The DSPy configuration for the module.
        """

        super().__init__()

        signature = ensure_signature(signature)  # type: ignore

        # Create a shallow copy of the tree_data
        self.tree_data = copy(tree_data)

        # Note which inputs are required
        self.message_update = message_update
        self.environment_level = environment_level
        self.collection_schemas = collection_schemas
        self.tasks_completed = tasks_completed
        self.collection_names = collection_names
        self.reasoning = reasoning
        self.impossible = impossible

        # == Inputs ==

        # -- User Prompt --
        user_prompt_desc = (
            "The user's original question/prompt that needs to be answered. "
            "This, possibly combined with the conversation history, will be used to determine your current action."
        )
        user_prompt_prefix = "${user_prompt}"
        user_prompt_field: str = dspy.InputField(
            prefix=user_prompt_prefix, desc=user_prompt_desc
        )

        # -- Conversation History --
        conversation_history_desc = (
            "Previous messages between user and assistant in chronological order: "
            "[{'role': 'user'|'assistant', 'content': str}] "
            "Use this to maintain conversation context and avoid repetition."
        )
        conversation_history_prefix = "${conversation_history}"
        conversation_history_field: list[dict] = dspy.InputField(
            prefix=conversation_history_prefix, desc=conversation_history_desc
        )

        # -- Atlas --
        atlas_desc = (
            "Your guide to how you should proceed as an agent in this task. "
            "This is pre-defined by the user."
        )
        atlas_prefix = "${atlas}"
        atlas_field: Atlas = dspy.InputField(prefix=atlas_prefix, desc=atlas_desc)

        # -- Errors --
        errors_desc = (
            "Any errors that have occurred during the previous attempt at this action. "
            "This is a list of dictionaries, containing details of the error. "
            "Make an attempt at providing different output to avoid this error now. "
            "If this error is repeated, or you judge it to be unsolvable, you can set `impossible` to True"
        )
        errors_prefix = "${previous_errors}"
        errors_field: list[dict] = dspy.InputField(
            prefix=errors_prefix, desc=errors_desc
        )

        # -- Add to Signature --
        extended_signature = signature.prepend(
            name="user_prompt", field=user_prompt_field, type_=str
        )
        extended_signature = extended_signature.append(
            name="conversation_history",
            field=conversation_history_field,
            type_=list[dict],
        )
        extended_signature = extended_signature.append(
            name="atlas", field=atlas_field, type_=Atlas
        )
        extended_signature = extended_signature.append(
            name="previous_errors", field=errors_field, type_=list[dict]
        )

        # == Optional Inputs / Outputs ==

        # -- Collection Schema Input --
        if collection_schemas:
            collection_schemas_desc = (
                "Metadata about available collections and their schemas: "
                "This is a dictionary with the following fields: "
                "{\n"
                "    name: collection name,\n"
                "    length: number of objects in the collection,\n"
                "    summary: summary of the collection,\n"
                "    fields: [\n"
                "        {\n"
                "            name: field_name,\n"
                "            groups: a dict with the value and count of each group.\n"
                "                a comprehensive list of all unique values that exist in the field.\n"
                "                if this is None, then no relevant groups were found.\n"
                "                these values are string, but the actual values in the collection are the 'type' of the field.\n"
                "            mean: mean of the field. if the field is text, this refers to the means length (in tokens) of the texts in this field. if the type is a list, this refers to the mean length of the lists,\n"
                "            range: minimum and maximum values of the length,\n"
                "            type: the data type of the field.\n"
                "        },\n"
                "        ...\n"
                "    ]\n"
                "}\n"
            )
            collection_schemas_prefix = "${collection_schemas}"
            collection_schemas_field: dict = dspy.InputField(
                prefix=collection_schemas_prefix, desc=collection_schemas_desc
            )
            extended_signature = extended_signature.append(
                name="collection_schemas", field=collection_schemas_field, type_=dict
            )

        # -- Tasks Completed Input --
        if tasks_completed:
            tasks_completed_desc = (
                "Which tasks have been completed in order. "
                "These are numbered so that higher numbers are more recent. "
                "Separated by prompts (so you should identify the prompt you are currently working on to see what tasks have been completed so far) "
                "Also includes reasoning for each task, to continue a decision logic across tasks. "
                "Use this to determine whether future searches, for this prompt are necessary, and what task(s) to choose. "
                "It is IMPORTANT that you separate what actions have been completed for which prompt, so you do not think you have failed an attempt for a different prompt."
            )
            tasks_completed_prefix = "${tasks_completed}"
            tasks_completed_field: str = dspy.InputField(
                prefix=tasks_completed_prefix, desc=tasks_completed_desc
            )
            extended_signature = extended_signature.append(
                name="tasks_completed", field=tasks_completed_field, type_=str
            )

        # -- Impossible Field --
        if impossible:
            impossible_desc = (
                "Given the actions you have available, and the environment/information. "
                "Is the task impossible to complete? "
                "I.e., do you wish that you had a different task to perform/choose from and hence should return to the base of the decision tree?"
                "Do not base this judgement on the entire prompt, as it is possible that other agents can perform other aspects of the request."
                "Do not judge impossibility based on if tasks have been completed, only on the current action and environment."
            )
            impossible_prefix = "${impossible}"
            impossible_field: bool = dspy.OutputField(
                prefix=impossible_prefix, desc=impossible_desc
            )
            extended_signature = extended_signature.prepend(
                name="impossible", field=impossible_field, type_=bool
            )

        # -- Message Update Output --
        if message_update:
            message_update_desc = (
                "Continue your current message to the user "
                "(latest assistant field in conversation history) with ONE concise sentence that: "
                "- Describes NEW technical details about your latest action "
                "- Highlights specific parameters or logic you just applied "
                "- Avoids repeating anything from conversation history "
                "- Speaks directly to them (no 'the user'), gender neutral message "
                "Just provide the new sentence update, not the full message from the conversation history. "
                "Your response should be based on only the part of the user's request that you can work on. "
                "It is possible other agents can perform other aspects of the request, so do not respond as if you cannot complete the entire request."
            )

            message_update_prefix = "${message_update}"
            message_update_field: str = dspy.OutputField(
                prefix=message_update_prefix, desc=message_update_desc
            )
            extended_signature = extended_signature.prepend(
                name="message_update", field=message_update_field, type_=str
            )

        # -- Reasoning Field --
        if reasoning:
            reasoning_desc = (
                "Reasoning: Repeat relevant parts of the any context within your environment, "
                "Evaluate all relevant information from the inputs, including any previous errors if applicable, "
                "use this to think step by step in order to answer the query."
                "Limit your reasoning to maximum 150 words. Only exceed this if the task is very complex. "
                "Your reasoning doubles up as communication to the user, so speak as if you are explaining tasks to them, "
                "presenting the logical sequence as messaging. "
                "Speak as if explaining your thoughts to a colleague, or an interviewer, who is monitoring how you solve problems. "
                "So do not say 'the user', speak directly to them. Use gender-neutral pronouns. "
            )
            reasoning_prefix = "${reasoning}"
            reasoning_field: str = dspy.OutputField(
                prefix=reasoning_prefix, desc=reasoning_desc
            )
            extended_signature = extended_signature.prepend(
                name="reasoning", field=reasoning_field, type_=str
            )

        # -- Predict --
        self.predict = dspy.Predict(extended_signature, **config)
        self.predict.signature.instructions += elysia_meta_prompt  # type: ignore

    def _add_environment_inputs(self, environment: Environment):

        if self.environment_level == "none":
            return

        if self.environment_level == "dynamic":

            if self.tree_data.view_env_vars is None:
                environment_json = environment._unhidden_to_json()
                use_metadata = (
                    estimate_tokens(json.dumps(environment_json))
                    > self.tree_data.env_token_limit
                )

            else:
                use_metadata = False

        else:
            use_metadata = self.environment_level == "metadata"

        # Check if environment metadata already exists in self.predict
        if use_metadata:
            if "environment_metadata" in self.predict.signature.fields:
                return
            else:

                if "environment" in self.predict.signature.fields:
                    self.predict.signature = self.predict.signature.delete(
                        "environment"
                    )

                environment_metadata_desc = (
                    "METADATA ONLY - Summary statistics about the environment, NOT the actual content. "
                    "Shows: tool names, object counts, metadata properties. "
                    "Does NOT show: actual message text, document content, specific data values. "
                    ""
                    "Format interpretation: "
                    "- Top level: tool name that added objects "
                    "- Each tool has X results "
                    "- Each result has Y objects with associated metadata properties "
                    ""
                    "IMPORTANT: This is like a table of contents - it tells you WHAT exists but not WHAT'S INSIDE. "
                )

                environment_metadata_prefix = "${environment_metadata}"
                environment_metadata_field: str = dspy.InputField(
                    description=environment_metadata_desc,
                    prefix=environment_metadata_prefix,
                    format=str,
                )
                self.predict.signature = self.predict.signature.append(
                    "environment_metadata", environment_metadata_field, str
                )
        else:
            if "environment" in self.predict.signature.fields:
                return
            else:

                if "environment_metadata" in self.predict.signature.fields:
                    self.predict.signature = self.predict.signature.delete(
                        "environment_metadata"
                    )

                environment_desc = (
                    "Information gathered from completed tasks. "
                    "Empty if no data has been retrieved yet. "
                    "Use to determine if more information is needed. "
                    "Additionally, use this as a reference to determine if you have already completed a task/what items are already available, to avoid repeating actions. "
                )
                environment_prefix = "${environment}"
                environment_field: dict = dspy.InputField(
                    prefix=environment_prefix, desc=environment_desc, format=dict
                )

                self.predict.signature = self.predict.signature.append(
                    name="environment", field=environment_field, type_=dict
                )

    def _add_tree_data_inputs(self, kwargs: dict) -> dict:

        self._add_environment_inputs(self.tree_data.environment)

        # Add the tree data inputs to the kwargs
        kwargs["user_prompt"] = self.tree_data.user_prompt
        kwargs["conversation_history"] = self.tree_data.conversation_history
        kwargs["atlas"] = self.tree_data.atlas
        kwargs["previous_errors"] = self.tree_data.get_errors()

        # Add the optional inputs to the kwargs
        if self.environment_level is not "none":
            if self.tree_data.view_env_vars is None:
                kwargs["environment"] = self.tree_data.environment.environment
            else:
                kwargs["environment"] = self.tree_data.environment._view(
                    self.tree_data.view_env_vars.tool_names,
                    self.tree_data.view_env_vars.metadata_keys,
                    self.tree_data.view_env_vars.metadata_values,
                )

            kwargs["environment_metadata"] = (
                self.tree_data.environment.output_llm_metadata()
            )

        if self.collection_schemas:
            if self.collection_names != []:
                kwargs["collection_schemas"] = (
                    self.tree_data.output_collection_metadata(
                        collection_names=self.collection_names, with_mappings=False
                    )
                )
            else:
                kwargs["collection_schemas"] = (
                    self.tree_data.output_collection_metadata(with_mappings=False)
                )

        if self.tasks_completed:
            kwargs["tasks_completed"] = self.tree_data.tasks_completed_string()

        return kwargs

    def forward(self, add_tree_data_inputs: bool = True, **kwargs):
        """
        Wrapper for the `.forward()` method of the signature provided on initialisation.
        Calls the LM synchronously.

        Args:
            add_tree_data_inputs (bool): Optional. Whether to add the tree data inputs to the kwargs.
                When enabled, this adds the inputs set up on initialisation (e.g. `message_update`, `reasoning`, etc).
                If disabled, only the normal inputs to the signature are passed (and the fields will be missing in the forward pass).
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to the signature (normal inputs to the signature).

        Returns:
            (dspy.Prediction): The DSPy prediction output from the LM call.
        """
        kwargs = self._add_tree_data_inputs(kwargs) if add_tree_data_inputs else kwargs
        return self.predict(**kwargs)

    async def aforward(self, add_tree_data_inputs: bool = True, **kwargs):
        """
        Wrapper for the `.aforward()` method of the signature provided on initialisation.
        Calls the LLM asynchronously.

        Args:
            add_tree_data_inputs (bool): Optional. Whether to add the tree data inputs to the kwargs.
                When enabled, this adds the inputs set up on initialisation (e.g. `message_update`, `reasoning`, etc).
                If disabled, only the normal inputs to the signature are passed (and the fields will be missing in the forward pass).
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to the signature.

        Returns:
            dspy.Prediction: The prediction from the signature.
        """
        kwargs = self._add_tree_data_inputs(kwargs) if add_tree_data_inputs else kwargs
        return await self.predict.acall(**kwargs)

    async def aforward_streaming(
        self, streamed_field: str, add_tree_data_inputs: bool = True, **kwargs
    ) -> AsyncGenerator[StreamResponse | dspy.Prediction, None]:
        """
        Performs an asynchronous forward pass to the signature with streaming enabled.

        Args:
            streamed_field (str): The name of the field to stream.
                If given as e.g. `"reasoning"`, then this method will yield `StreamResponse`s whose `chunk` attributes are text pieces of the reasoning field.
                The field given must be a string output type, otherwise this will error.
            add_tree_data_inputs (bool): Optional. Whether to add the tree data inputs to the kwargs.
                When enabled, this adds the inputs set up on initialisation (e.g. `message_update`, `reasoning`, etc).
                If disabled, only the normal inputs to the signature are passed (and the fields will be missing in the forward pass).
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to the signature.

        Returns:
            AsyncGenerator[StreamResponse | dspy.Prediction, None]: The prediction from the signature.
        """

        kwargs = self._add_tree_data_inputs(kwargs) if add_tree_data_inputs else kwargs
        stream_predict = dspy.streamify(
            self.predict,
            stream_listeners=[StreamListener(signature_field_name=streamed_field)],
            is_async_program=True,
            async_streaming=True,
        )
        output_stream = stream_predict(**kwargs)
        async for chunk in output_stream:
            yield chunk

    async def aforward_with_feedback_examples(
        self,
        feedback_model: str,
        client_manager: ClientManager,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        add_tree_data_inputs: bool = True,
        **kwargs,
    ) -> tuple[dspy.Prediction, list[str]] | dspy.Prediction:
        """
        Performs an asynchronous forward pass to the signature with few-shot learning via feedback examples.
        This requires connection to a Weaviate instance (via the `client_manager`) to retrieve examples from a collection.

        This will first retrieve examples from the feedback collection, and use those as few-shot examples to run the module.
        It retrieves based from vectorising and searching on the user's prompt, finding similar prompts from the feedback collection.
        This is an EXPERIMENTAL feature, and may not work as expected.

        If the number of examples is less than `num_base_lm_examples`, the module will use the complex LM.
        Otherwise, it will use the base LM. This is so that the less accurate, but faster base LM can be used when guidance is available.
        However, when there are insufficient examples, the complex LM will be used.

        Args:
            feedback_model (str): The label of the feedback data to use as examples.
                E.g., "decision" is the default name given to examples for the LM in the decision tree.
                This is used to retrieve the examples from the feedback collection.
            client_manager (ClientManager): The client manager to use.
            base_lm (dspy.LM): The base LM to (conditionally) use.
            complex_lm (dspy.LM): The complex LM to (conditionally) use.
            **kwargs (Any): The keyword arguments to pass to the forward pass.
                Important: All additional inputs to the DSPy module should be passed here as keyword arguments.
                Also: Do not include `lm` in the kwargs, as this will be set automatically.

        Returns:
            (dspy.Prediction): The prediction from the forward pass.
        """

        examples, uuids = await retrieve_feedback(
            client_manager,
            self.tree_data.user_prompt,
            feedback_model,
            user_id=self.tree_data.user_id,
            n=10,
        )
        if len(examples) > 0:
            optimizer = dspy.LabeledFewShot(k=10)
            optimized_module = optimizer.compile(self, trainset=examples)
        else:
            return (
                await self.aforward(
                    lm=complex_lm,
                    add_tree_data_inputs=add_tree_data_inputs,
                    **kwargs,
                ),
                uuids,
            )

        # Select the LM to use based on the number of examples
        if len(examples) < self.tree_data.settings.NUM_FEEDBACK_EXAMPLES:
            return (
                await optimized_module.aforward(
                    lm=complex_lm,
                    add_tree_data_inputs=add_tree_data_inputs,
                    **kwargs,
                ),
                uuids,
            )
        else:
            return (
                await optimized_module.aforward(
                    lm=base_lm,
                    add_tree_data_inputs=add_tree_data_inputs,
                    **kwargs,
                ),
                uuids,
            )

    async def aforward_streaming_with_feedback_examples(
        self,
        streamed_field: str,
        feedback_model: str,
        client_manager: ClientManager,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        add_tree_data_inputs: bool = True,
        **kwargs,
    ) -> AsyncGenerator[dspy.Prediction | list[str] | StreamResponse, None]:
        """
        Performs an asynchronous forward pass to the signature with few-shot learning via feedback examples AND streaming.
        Same as `aforward_with_feedback_examples` but is an async generator function.

        Args:
            streamed_field (str): The field in the dspy Module that will be streamed (must be a string output field).
            feedback_model (str): The label of the feedback data to use as examples.
                E.g., "decision" is the default name given to examples for the LM in the decision tree.
                This is used to retrieve the examples from the feedback collection.
            client_manager (ClientManager): The client manager to use.
            base_lm (dspy.LM): The base LM to (conditionally) use.
            complex_lm (dspy.LM): The complex LM to (conditionally) use.
            **kwargs (Any): The keyword arguments to pass to the forward pass.
                Important: All additional inputs to the DSPy module should be passed here as keyword arguments.
                Also: Do not include `lm` in the kwargs, as this will be set automatically.

        Yields:
            (dspy.Prediction): The prediction from the forward pass.
            (list): the UUIDs used as few-shot examples retrieved from the database.
            (StreamResponse): The chunks from the streaming for the given streamed_field.
        """
        examples, uuids = await retrieve_feedback(
            client_manager,
            self.tree_data.user_prompt,
            feedback_model,
            user_id=self.tree_data.user_id,
            n=10,
        )
        yield uuids

        # No examples - use complex LM directly
        if not examples:
            async for result in self.aforward_streaming(
                streamed_field,
                lm=complex_lm,
                add_tree_data_inputs=add_tree_data_inputs,
                **kwargs,
            ):
                yield result
            return

        # Optimize with few-shot examples
        optimized_module = dspy.LabeledFewShot(k=10).compile(self, trainset=examples)

        # Choose LM based on example count
        lm = (
            base_lm
            if len(examples) >= self.tree_data.settings.NUM_FEEDBACK_EXAMPLES
            else complex_lm
        )
        async for result in optimized_module.aforward_streaming(
            streamed_field, lm=lm, add_tree_data_inputs=add_tree_data_inputs, **kwargs
        ):
            yield result
