from dspy import LM, configure, ChatAdapter
from dspy.utils import DummyLM
from contextlib import contextmanager
from pydantic_core import PydanticUndefined


class DummyAdapter(ChatAdapter):

    def __init__(
        self, callbacks: list | None = None, use_native_function_calling: bool = True
    ):
        super().__init__(
            callbacks=callbacks, use_native_function_calling=use_native_function_calling
        )

    def __call__(
        self,
        lm: LM,
        lm_kwargs,
        signature,
        demos,
        inputs,
    ):

        categories = []
        defaults = {}

        # set some specific defaults for different inputs

        # for decision node
        if "available_actions" in inputs:
            defaults["function_name"] = list(inputs["available_actions"].keys())[0]

        # for preprocessing return type assignment
        if "possible_return_types" in inputs:
            defaults["return_types"] = [list(inputs["possible_return_types"].keys())[0]]

        # for field mapping
        if "input_data_fields" in inputs and "output_data_fields" in inputs:
            defaults["field_mapping"] = {
                in_field: inputs["output_data_fields"][0]
                for in_field in inputs["input_data_fields"]
            }

        # otherwise, try to find the default value from the signature to ensure type checking
        for field_name, field in signature.model_fields.items():
            if field.json_schema_extra.get("__dspy_field_type") == "output":

                categories.append(field_name)

                if field_name not in defaults:
                    if field.default is not PydanticUndefined:
                        defaults[field_name] = field.default
                    else:
                        try:
                            defaults[field_name] = field.annotation()
                        except Exception as e:
                            try:
                                defaults[field_name] = field.annotation()
                            except Exception as e:
                                try:
                                    defaults[field_name] = field.json_schema_extra[
                                        "format"
                                    ]()
                                except Exception as e:
                                    defaults[field_name] = "Test!"

        return super().__call__(
            DummyLM([{cat: defaults[cat] for cat in categories}]),
            lm_kwargs,
            signature,
            demos,
            inputs,
        )

    async def acall(
        self,
        lm: LM,
        lm_kwargs,
        signature,
        demos,
        inputs,
    ):

        categories = []
        defaults = {}

        # set some specific defaults for different inputs

        # for decision node
        if "available_actions" in inputs:
            defaults["function_name"] = list(inputs["available_actions"].keys())[0]

        # for preprocessing return type assignment
        if "possible_return_types" in inputs:
            defaults["return_types"] = [list(inputs["possible_return_types"].keys())[0]]

        # for field mapping
        if "input_data_fields" in inputs and "output_data_fields" in inputs:
            defaults["field_mapping"] = {
                in_field: inputs["output_data_fields"][0]
                for in_field in inputs["input_data_fields"]
            }

        # otherwise, try to find the default value from the signature to ensure type checking
        for field_name, field in signature.model_fields.items():
            if field.json_schema_extra.get("__dspy_field_type") == "output":

                categories.append(field_name)

                if field_name not in defaults:
                    if field.default is not PydanticUndefined:
                        defaults[field_name] = field.default
                    else:
                        try:
                            defaults[field_name] = field.annotation()
                        except Exception as e:
                            try:
                                defaults[field_name] = field.annotation()
                            except Exception as e:
                                try:
                                    defaults[field_name] = field.json_schema_extra[
                                        "format"
                                    ]()
                                except Exception as e:
                                    defaults[field_name] = "Test!"

        return await super().acall(
            DummyLM([{cat: defaults[cat] for cat in categories}]),
            lm_kwargs,
            signature,
            demos,
            inputs,
        )
