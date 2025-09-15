import os
import logging
import litellm
from litellm import (
    AuthenticationError,
    NotFoundError,
    BadRequestError,
    models_by_provider,
)
from litellm.utils import get_valid_models, check_valid_key
from rich.logging import RichHandler
from typing import Callable, Literal

import spacy
import random

from dotenv import load_dotenv
from dspy import LM
from copy import deepcopy

load_dotenv(override=True)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")  # type: ignore
    nlp = spacy.load("en_core_web_sm")


api_key_to_provider = {
    "openai_api_key": ["openai"],
    "anthropic_api_key": ["anthropic"],
    "openrouter_api_key": [
        "openrouter/openai",
        "openrouter/anthropic",
        "openrouter/google",
    ],
    "gemini_api_key": ["gemini"],
    "azure_api_key": ["azure"],
}

provider_to_models = {
    "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "gpt-4o"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
    ],
    "openrouter/openai": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "gpt-4o",
    ],
    "openrouter/anthropic": [
        "claude-sonnet-4",
        "claude-3.7-sonnet",
        "claude-3.5-haiku",
    ],
    "openrouter/google": [
        "gemini-2.5-flash",
        "gemini-2.0-flash-001",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.0-flash-001",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    ],
}

provider_to_api_keys = {
    "openai": ["openai_api_key"],
    "anthropic": ["anthropic_api_key"],
    "openrouter/openai": ["openrouter_api_key"],
    "openrouter/anthropic": ["openrouter_api_key"],
    "openrouter/google": ["openrouter_api_key"],
    "gemini": ["gemini_api_key"],
    "groq": ["groq_api_key"],
    "databricks": ["databricks_api_key", "databricks_api_base"],
    "bedrock": ["aws_access_key_id", "aws_secret_access_key", "aws_region_name"],
    "snowflake": ["snowflake_jwt", "snowflake_account_id"],
}


def get_available_models(api_keys: list[str]):
    available_models = []
    for api_key in api_keys:
        if api_key in api_key_to_provider:
            for provider in api_key_to_provider[api_key]:
                if provider in provider_to_models:
                    available_models.extend(provider_to_models[provider])
    return list(set(available_models))


def get_available_providers(api_keys: list[str]) -> list[str]:
    available_providers = []
    for api_key in api_keys:
        if api_key in api_key_to_provider:
            available_providers.extend(api_key_to_provider[api_key])
    return list(set(available_providers))


def is_api_key(key: str) -> bool:
    return (
        key.lower().endswith("api_key")
        or key.lower().endswith("apikey")
        or key.lower().endswith("api_version")
        or key.lower().endswith("api_base")
        or key.lower().endswith("_account_id")
        or key.lower().endswith("_jwt")
        or key.lower().endswith("_access_key_id")
        or key.lower().endswith("_secret_access_key")
        or key.lower().endswith("_region_name")
    )


class Settings:
    """
    Settings for Elysia.
    This class handles the configuration of various settings within Elysia.
    This includes:
    - The base and complex models to use.
    - The providers for the base and complex models.
    - The Weaviate cloud URL and API key.
    - The API keys for the providers.
    - The logger and logging level.

    The Settings object is set as at a default `settings` object if running Elysia as a package.
    You can import this via `elysia.config.settings`.
    This is initialised to default values from the environment variables.

    Or, you can create your own Settings object, and configure it as you wish.
    The Settings object can be passed to different Elysia classes and functions, such as `Tree` and `preprocess`.
    These will not use the global `settings` object, but instead use the Settings object you passed to them.
    """

    def __init__(self):
        """
        Initialize the settings for Elysia.
        These are all settings initialised to None, and should be set using the `configure` method.
        """
        # Default settings
        self.SETTINGS_ID = str(random.randint(100000000000000, 999999999999999))

        self.base_init()

    def base_init(self):
        self.BASE_MODEL: str | None = None
        self.BASE_PROVIDER: str | None = None
        self.COMPLEX_MODEL: str | None = None
        self.COMPLEX_PROVIDER: str | None = None

        self.WCD_URL: str = ""
        self.WCD_API_KEY: str = ""

        self.MODEL_API_BASE: str | None = None

        self.API_KEYS: dict[str, str] = {}

        self.logger = logging.getLogger("rich")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers before adding a new one
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))

        self.logger.propagate = False
        self.LOGGING_LEVEL = "INFO"
        self.LOGGING_LEVEL_INT = 20

        # Experimental features
        self.USE_FEEDBACK = False
        self.BASE_USE_REASONING = True
        self.COMPLEX_USE_REASONING = True

    def setup_app_logger(self, logger: logging.Logger):
        """
        Override existing logger with the app-level logger.

        Args:
            logger (Logger): The logger to use.
        """
        self.logger = logger
        self.LOGGING_LEVEL_INT = logger.level
        inverted_logging_mapping = {
            v: k for k, v in logging.getLevelNamesMapping().items()
        }
        self.LOGGING_LEVEL = inverted_logging_mapping[self.LOGGING_LEVEL_INT]

    def configure_logger(
        self,
        level: Literal[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"
        ] = "NOTSET",
    ):
        """
        Configure the logger with a RichHandler.

        Args:
            level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]): The logging level to use.
        """
        self.logger.setLevel(level)
        self.LOGGING_LEVEL = level
        self.LOGGING_LEVEL_INT = logging.getLevelNamesMapping()[level]

    def set_api_key(self, api_key: str, api_key_name: str) -> None:
        self.API_KEYS[api_key_name] = api_key

    def get_api_key(self, api_key_name: str) -> str:
        return self.API_KEYS[api_key_name]

    def load_settings(self, settings: dict):
        for item in settings:
            setattr(self, item, settings[item])

        # self.logger = logging.getLogger("rich")
        # self.logger.setLevel(self.LOGGING_LEVEL)
        # for handler in self.logger.handlers[:]:
        #     self.logger.removeHandler(handler)
        # self.logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))

    @classmethod
    def from_smart_setup(cls):
        settings = cls()
        settings.set_from_env()
        settings.smart_setup()
        return settings

    @classmethod
    def from_env_vars(cls):
        settings = cls()
        settings.set_from_env()
        return settings

    def set_from_env(self):
        self.BASE_MODEL = os.getenv("BASE_MODEL", None)
        self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", None)
        self.BASE_PROVIDER = os.getenv("BASE_PROVIDER", None)
        self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER", None)
        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "NOTSET")
        self.set_api_keys_from_env()

    def set_api_keys_from_env(self):

        self.WCD_URL = os.getenv(
            "WEAVIATE_URL",
            os.getenv("WCD_URL", ""),
        )
        self.WCD_API_KEY = os.getenv(
            "WEAVIATE_API_KEY",
            os.getenv("WCD_API_KEY", ""),
        )

        self.API_KEYS = {
            env_var.lower(): os.getenv(env_var, "")
            for env_var in os.environ
            if is_api_key(env_var) and env_var.lower() != "wcd_api_key"
        }
        for api_key in self.API_KEYS:
            self.set_api_key(self.API_KEYS[api_key], api_key)

    def smart_setup(self):

        # Check if the user has set the base model etc from the environment variables
        if os.getenv("BASE_MODEL", None):
            self.BASE_MODEL = os.getenv("BASE_MODEL")
        if os.getenv("COMPLEX_MODEL", None):
            self.COMPLEX_MODEL = os.getenv("COMPLEX_MODEL")
        if os.getenv("BASE_PROVIDER", None):
            self.BASE_PROVIDER = os.getenv("BASE_PROVIDER")
        if os.getenv("COMPLEX_PROVIDER", None):
            self.COMPLEX_PROVIDER = os.getenv("COMPLEX_PROVIDER")

        self.MODEL_API_BASE = os.getenv("MODEL_API_BASE", None)
        self.LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "NOTSET")

        self.set_api_keys_from_env()

        # check what API keys are available
        if (
            self.BASE_MODEL is None
            or self.COMPLEX_MODEL is None
            or self.BASE_PROVIDER is None
            or self.COMPLEX_PROVIDER is None
        ):
            if os.getenv("OPENROUTER_API_KEY", None):
                # use gemini 2.0 flash
                self.BASE_PROVIDER = "openrouter/google"
                self.COMPLEX_PROVIDER = "openrouter/google"
                self.BASE_MODEL = "gemini-2.0-flash-001"
                self.COMPLEX_MODEL = "gemini-2.5-flash"
            elif os.getenv("GEMINI_API_KEY", None):
                # use gemini 2.0 flash
                self.BASE_PROVIDER = "gemini"
                self.COMPLEX_PROVIDER = "gemini"
                self.BASE_MODEL = "gemini-2.0-flash-001"
                self.COMPLEX_MODEL = "gemini-2.5-flash"
            elif os.getenv("OPENAI_API_KEY", None):
                # use gpt family
                self.BASE_PROVIDER = "openai"
                self.COMPLEX_PROVIDER = "openai"
                self.BASE_MODEL = "gpt-4.1-mini"
                self.COMPLEX_MODEL = "gpt-4.1"
            elif os.getenv("ANTHROPIC_API_KEY", None):
                # use claude family
                self.BASE_PROVIDER = "anthropic"
                self.COMPLEX_PROVIDER = "anthropic"
                self.BASE_MODEL = "claude-3-5-haiku-latest"
                self.COMPLEX_MODEL = "claude-sonnet-4-0"

    def configure(
        self,
        replace: bool = False,
        **kwargs,
    ):
        """
        Configure the settings for Elysia for the current Settings object.

        Args:
            replace (bool): Whether to override the current settings with the new settings.
                When this is True, all existing settings are removed, and only the new settings are used.
                Defaults to False.
            **kwargs (str): One or more of the following:
                - base_model (str): The base model to use. e.g. "gpt-4o-mini"
                - complex_model (str): The complex model to use. e.g. "gpt-4o"
                - base_provider (str): The provider to use for base_model. E.g. "openai" or "openrouter/openai"
                - complex_provider (str): The provider to use for complex_model. E.g. "openai" or "openrouter/openai"
                - model_api_base (str): The API base to use.
                - wcd_url (str): The Weaviate cloud URL to use.
                - wcd_api_key (str): The Weaviate cloud API key to use.
                - logging_level (str): The logging level to use. e.g. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
                - use_feedback (bool): EXPERIMENTAL. Whether to use feedback from previous runs of the tree.
                    If True, the tree will use TrainingUpdate objects that have been saved in previous runs of the decision tree.
                    These are implemented via few-shot examples for the decision node.
                    They are collected in the 'feedback' collection (ELYSIA_FEEDBACK__).
                    Relevant examples are retrieved from the collection based on searching the collection via the user's prompt.
                - base_use_reasoning (bool): Whether to use reasoning output for the base model.
                    If True, the model will generate reasoning before coming to its solution.
                - complex_use_reasoning (bool): Whether to use reasoning output for the complex model.
                    If True, the model will generate reasoning before coming to its solution.
                - Additional API keys to set. E.g. `openai_apikey="..."`, if this argument ends with `apikey` or `api_key`,
                    it will be added to the `API_KEYS` dictionary.

        """
        if replace:
            self.base_init()

        # convert all kwargs to lowercase for consistency
        kwargs = {kwarg.lower(): kwargs[kwarg] for kwarg in kwargs}

        if "base_model" in kwargs:
            if "base_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if base_model is set. "
                    "E.g. `elysia.config.configure(base_model='gpt-4o-mini', base_provider='openai')`"
                )

            if kwargs["base_provider"] == "ollama" and (
                not kwargs["model_api_base"] or "MODEL_API_BASE" not in dir(self)
            ):
                raise ValueError(
                    "Using local models via ollama requires MODEL_API_BASE to be set. "
                    "This is likely to be http://localhost:11434. "
                    "e.g. `elysia.config.configure(model_api_base='http://localhost:11434')`"
                )

            self.BASE_MODEL = kwargs["base_model"]
            self.BASE_PROVIDER = kwargs["base_provider"]

            kwargs.pop("base_model")
            kwargs.pop("base_provider")

            # self.load_base_dspy_model()

        if "complex_model" in kwargs:
            if "complex_provider" not in kwargs:
                raise ValueError(
                    "Provider must be specified if complex_model is set. "
                    "E.g. `elysia.config.configure(complex_model='gpt-4o', complex_provider='openai')`"
                )

            if kwargs["complex_provider"] == "ollama" and (
                not kwargs["model_api_base"] or "MODEL_API_BASE" not in dir(self)
            ):
                raise ValueError(
                    "Using local models via ollama requires MODEL_API_BASE to be set. "
                    "This is likely to be http://localhost:11434. "
                    "e.g. `elysia.config.configure(model_api_base='http://localhost:11434')`"
                )

            self.COMPLEX_MODEL = kwargs["complex_model"]
            self.COMPLEX_PROVIDER = kwargs["complex_provider"]

            kwargs.pop("complex_model")
            kwargs.pop("complex_provider")

            # self.load_complex_dspy_model()

        if "model_api_base" in kwargs:
            self.MODEL_API_BASE = kwargs["model_api_base"]
            kwargs.pop("model_api_base")

        if "wcd_url" in kwargs:
            self.WCD_URL = kwargs["wcd_url"]
            kwargs.pop("wcd_url")

        if "wcd_api_key" in kwargs:
            self.WCD_API_KEY = kwargs["wcd_api_key"]
            kwargs.pop("wcd_api_key")

        if "weaviate_url" in kwargs:
            self.WCD_URL = kwargs["weaviate_url"]
            kwargs.pop("weaviate_url")

        if "weaviate_api_key" in kwargs:
            self.WCD_API_KEY = kwargs["weaviate_api_key"]
            kwargs.pop("weaviate_api_key")

        if "logging_level" in kwargs or "logger_level" in kwargs:

            self.LOGGING_LEVEL = (
                kwargs["logging_level"]
                if "logging_level" in kwargs
                else kwargs["logger_level"]
            )
            self.LOGGING_LEVEL_INT = logging.getLevelNamesMapping()[self.LOGGING_LEVEL]
            self.logger.setLevel(self.LOGGING_LEVEL)
            if "logging_level" in kwargs:
                kwargs.pop("logging_level")
            if "logger_level" in kwargs:
                kwargs.pop("logger_level")
            if "logging_level_int" in kwargs:
                kwargs.pop("logging_level_int")
            if "logger_level_int" in kwargs:
                kwargs.pop("logger_level_int")

        if "logging_level_int" in kwargs or "logger_level_int" in kwargs:

            self.LOGGING_LEVEL_INT = (
                kwargs["logging_level_int"]
                if "logging_level_int" in kwargs
                else kwargs["logger_level_int"]
            )
            self.LOGGING_LEVEL = {
                v: k for k, v in logging.getLevelNamesMapping().items()
            }[self.LOGGING_LEVEL_INT]
            self.logger.setLevel(self.LOGGING_LEVEL)

        if "settings_id" in kwargs:
            self.SETTINGS_ID = kwargs["settings_id"]
            kwargs.pop("settings_id")

        if "use_feedback" in kwargs:
            self.USE_FEEDBACK = kwargs["use_feedback"]
            kwargs.pop("use_feedback")

        if "base_use_reasoning" in kwargs:
            self.BASE_USE_REASONING = kwargs["base_use_reasoning"]
            kwargs.pop("base_use_reasoning")

        if "complex_use_reasoning" in kwargs:
            self.COMPLEX_USE_REASONING = kwargs["complex_use_reasoning"]
            kwargs.pop("complex_use_reasoning")

        if "api_keys" in kwargs and isinstance(kwargs["api_keys"], dict):
            for key, value in kwargs["api_keys"].items():
                self.set_api_key(value, key)
            kwargs.pop("api_keys")

        # remainder of kwargs are API keys or saved there
        removed_kwargs = []
        for key, value in kwargs.items():
            if is_api_key(key):
                self.set_api_key(value, key)
                removed_kwargs.append(key)

        for key in removed_kwargs:
            kwargs.pop(key)

        # remaining kwargs are not API keys
        if len(kwargs) > 0:
            self.logger.warning(
                "Unknown arguments to configure: " + ", ".join(kwargs.keys())
            )

    def __repr__(self) -> str:
        out = ""
        if "BASE_MODEL" in dir(self) and self.BASE_MODEL is not None:
            out += f"Base model: {self.BASE_MODEL}\n"
        else:
            out += "Base model: not set\n"
        if "COMPLEX_MODEL" in dir(self) and self.COMPLEX_MODEL is not None:
            out += f"Complex model: {self.COMPLEX_MODEL}\n"
        else:
            out += "Complex model: not set\n"
        if "BASE_PROVIDER" in dir(self) and self.BASE_PROVIDER is not None:
            out += f"Base provider: {self.BASE_PROVIDER}\n"
        else:
            out += "Base provider: not set\n"
        if "COMPLEX_PROVIDER" in dir(self) and self.COMPLEX_PROVIDER is not None:
            out += f"Complex provider: {self.COMPLEX_PROVIDER}\n"
        else:
            out += "Complex provider: not set\n"
        if "MODEL_API_BASE" in dir(self):
            out += f"Model API base: {self.MODEL_API_BASE}\n"
        else:
            out += "Model API base: not set\n"
        return out

    def to_json(self):
        return {
            item: getattr(self, item)
            for item in dir(self)
            if not item.startswith("_")
            and not isinstance(getattr(self, item), Callable)
            and item not in ["BASE_MODEL_LM", "COMPLEX_MODEL_LM", "logger"]
        }

    @classmethod
    def from_json(cls, json_data: dict):
        settings = cls()
        for item in json_data:
            if item not in ["logger"]:
                setattr(settings, item, json_data[item])

        settings.logger.setLevel(settings.LOGGING_LEVEL)

        return settings

    def check(self):
        return {
            "base_model": self.BASE_MODEL is not None and self.BASE_MODEL != "",
            "base_provider": self.BASE_PROVIDER is not None
            and self.BASE_PROVIDER != "",
            "complex_model": self.COMPLEX_MODEL is not None
            and self.COMPLEX_MODEL != "",
            "complex_provider": self.COMPLEX_PROVIDER is not None
            and self.COMPLEX_PROVIDER != "",
            "wcd_url": self.WCD_URL != "",
            "wcd_api_key": self.WCD_API_KEY != "",
        }


class APIKeyError(Exception):
    pass


class IncorrectModelError(Exception):
    pass


class ElysiaKeyManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _check_model_availability(self, model: str, provider: str) -> bool:
        if provider not in models_by_provider and not provider.startswith("openrouter"):
            raise IncorrectModelError(
                f"The provider {provider} is not available. "
                f"Some example providers are: {', '.join(list(models_by_provider.keys())[:5])}. "
                "Check the full model documentation (https://docs.litellm.ai/docs/providers)."
            )

        elif (
            provider.startswith("openrouter/")
            and f"{provider}/{model}" not in models_by_provider["openrouter"]
        ):
            example_models = [
                model.split("/")[-1]
                for model in models_by_provider["openrouter"]
                if model.startswith(provider)
            ]

            raise IncorrectModelError(
                f"The model {model} is not available for provider {provider}. "
                f"Some example models for {provider} are: "
                f"{', '.join(example_models[:5])}. "
                "Check the full model documentation (https://docs.litellm.ai/docs/providers)."
            )

        elif (
            provider in models_by_provider and model not in models_by_provider[provider]
        ):
            raise IncorrectModelError(
                f"The model {model} is not available for provider {provider}. "
                f"Some example models for {provider} are: "
                f"{', '.join(models_by_provider[provider][:5])}. "
                "Check the full model documentation (https://docs.litellm.ai/docs/providers)."
            )

        return False

    def __enter__(self):

        # save existing env
        self.existing_env = deepcopy(os.environ)

        # blank out the environ except for certain values
        os.environ = {
            "MODEL_API_BASE": self.settings.MODEL_API_BASE,
            "WCD_URL": self.settings.WCD_URL,
            "WCD_API_KEY": self.settings.WCD_API_KEY,
        }

        # update all api keys in env
        for api_key, value in self.settings.API_KEYS.items():
            os.environ[api_key.upper()] = value

        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # reset env to original
        os.environ = self.existing_env

        if exc_type is NotFoundError or exc_type is BadRequestError:
            self._check_model_availability(
                self.settings.BASE_MODEL, self.settings.BASE_PROVIDER
            )
            self._check_model_availability(
                self.settings.COMPLEX_MODEL, self.settings.COMPLEX_PROVIDER
            )

        if exc_type is AuthenticationError or exc_type is BadRequestError:
            missing_base_api_keys = [
                api_key
                for api_key in provider_to_api_keys[self.settings.BASE_PROVIDER]
                if api_key not in self.settings.API_KEYS
            ]
            missing_complex_api_keys = [
                api_key
                for api_key in provider_to_api_keys[self.settings.COMPLEX_PROVIDER]
                if api_key not in self.settings.API_KEYS
            ]
            if len(missing_base_api_keys) > 0:
                raise APIKeyError(
                    f"You are trying to use the model '{self.settings.BASE_MODEL}' "
                    f"but you do not have one of the following API keys: {', '.join(missing_base_api_keys)}. "
                    f"Please update your API keys in the settings."
                )
            if len(missing_complex_api_keys) > 0:
                raise APIKeyError(
                    f"You are trying to use the model '{self.settings.COMPLEX_MODEL}' "
                    f"but you do not have one of the following API keys: {', '.join(missing_complex_api_keys)}. "
                    f"Please update your API keys in the settings."
                )

            raise APIKeyError(
                f"One of your API keys is incorrect. "
                f"Please update your API keys in the settings. "
                f"The relevant API keys are: "
                f"{set(provider_to_api_keys[self.settings.BASE_PROVIDER] + provider_to_api_keys[self.settings.COMPLEX_PROVIDER])}"
            )

        if (
            exc_type is AuthenticationError
            or exc_type is BadRequestError
            or exc_type is NotFoundError
        ):
            raise IncorrectModelError(
                f"Either one of the models or providers: '{self.settings.BASE_MODEL}' or '{self.settings.COMPLEX_MODEL}' "
                f"({self.settings.BASE_PROVIDER} or {self.settings.COMPLEX_PROVIDER}) "
                f"is incorrect, or you do not have the required API keys. "
                f"Check the model documentation (https://docs.litellm.ai/docs/providers)."
            )

        pass


def check_base_lm_settings(settings: Settings):
    if "BASE_MODEL" not in dir(settings) or settings.BASE_MODEL is None:
        available_models = get_available_models(list(settings.API_KEYS.keys()))
        raise IncorrectModelError(
            f"No base model specified. "
            f"Available models based on your API keys: {', '.join(available_models)}"
        )

    if "BASE_PROVIDER" not in dir(settings) or settings.BASE_PROVIDER is None:
        available_providers = get_available_providers(list(settings.API_KEYS.keys()))
        raise IncorrectModelError(
            f"No base provider specified. "
            f"Available providers based on your API keys: {', '.join(available_providers)}"
        )


def check_complex_lm_settings(settings: Settings):
    if "COMPLEX_MODEL" not in dir(settings) or settings.COMPLEX_MODEL is None:
        available_models = get_available_models(list(settings.API_KEYS.keys()))
        raise IncorrectModelError(
            f"No complex model specified. "
            f"Available models based on your API keys: {', '.join(available_models)}"
        )

    if "COMPLEX_PROVIDER" not in dir(settings) or settings.COMPLEX_PROVIDER is None:
        available_providers = get_available_providers(list(settings.API_KEYS.keys()))
        raise IncorrectModelError(
            f"No complex provider specified. "
            f"Available providers based on your API keys: {', '.join(available_providers)}"
        )


def load_base_lm(settings: Settings) -> LM:
    check_base_lm_settings(settings)

    return load_lm(
        settings.BASE_PROVIDER,
        settings.BASE_MODEL,
        settings.MODEL_API_BASE if "MODEL_API_BASE" in dir(settings) else None,
    )


def load_complex_lm(settings: Settings) -> LM:
    check_complex_lm_settings(settings)

    return load_lm(
        settings.COMPLEX_PROVIDER,
        settings.COMPLEX_MODEL,
        settings.MODEL_API_BASE if "MODEL_API_BASE" in dir(settings) else None,
    )


def load_lm(
    provider: str | None,
    lm_name: str | None,
    model_api_base: str | None = None,
) -> LM:

    if provider is None or lm_name is None:
        raise ValueError("Provider and LM name must be set")

    api_base = model_api_base if provider == "ollama" else None
    full_lm_name = f"{provider}/{lm_name}"

    if lm_name.startswith("o1") or lm_name.startswith("o3"):
        return LM(
            model=full_lm_name, api_base=api_base, max_tokens=8000, temperature=1.0
        )

    return LM(model=full_lm_name, api_base=api_base, max_tokens=8000)


# global settings that should never be used by the frontend
# but used when using Elysia as a package
settings = Settings()
settings.smart_setup()

DEFAULT_SETTINGS = Settings()
DEFAULT_SETTINGS.smart_setup()


def smart_setup() -> None:
    settings.smart_setup()


def set_from_env() -> None:
    settings.set_from_env()


def reset_settings() -> None:
    settings.base_init()
    settings.smart_setup()


def configure(**kwargs) -> None:
    """
    Configure the settings for Elysia for the global settings object.

    Args:
        **kwargs (str): One or more of the following:
            - base_model (str): The base model to use. e.g. "gpt-4o-mini"
            - complex_model (str): The complex model to use. e.g. "gpt-4o"
            - base_provider (str): The provider to use for base_model. E.g. "openai" or "openrouter/openai"
            - complex_provider (str): The provider to use for complex_model. E.g. "openai" or "openrouter/openai"
            - model_api_base (str): The API base to use.
            - wcd_url (str): The Weaviate cloud URL to use.
            - wcd_api_key (str): The Weaviate cloud API key to use.
            - logging_level (str): The logging level to use. e.g. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            - use_feedback (bool): EXPERIMENTAL. Whether to use feedback from previous runs of the tree.
                If True, the tree will use TrainingUpdate objects that have been saved in previous runs of the decision tree.
                These are implemented via few-shot examples for the decision node.
                They are collected in the 'feedback' collection (ELYSIA_FEEDBACK__).
                Relevant examples are retrieved from the collection based on searching the collection via the user's prompt.
            - Additional API keys to set. E.g. `openai_apikey="..."`, if this argument ends with `apikey` or `api_key`,
                it will be added to the `API_KEYS` dictionary.

    """
    settings.configure(**kwargs)
