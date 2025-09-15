import dspy
import pytest
from elysia.config import Settings, configure, reset_settings, IncorrectModelError
from elysia.tree.tree import Tree


class TestTreeConfig:

    def reset_global_settings(self):
        reset_settings()

    # def test_change_llm_in_tree_global(self):
    #     """
    #     When using global settings, is the change reflected in the tree?
    #     """
    #     self.reset_global_settings()

    #     # create a Tree (uses global settings by default)
    #     tree = Tree(low_memory=True)

    #     # should raise an error, no models set
    #     with pytest.raises(ValueError):
    #         tree.get_lm("base")
    #         tree.get_lm("complex")
    #     assert tree.base_lm is None
    #     assert tree.complex_lm is None

    #     # change the base model
    #     configure(base_model="gpt-4o-mini", base_provider="openai")

    #     # should now be changed (no error)
    #     base_lm_loaded_in_tree = tree.get_lm("base")  # should be a dspy.LM

    #     assert isinstance(base_lm_loaded_in_tree, dspy.LM)
    #     assert base_lm_loaded_in_tree.model == "openai/gpt-4o-mini"
    #     assert tree.settings.BASE_MODEL == "gpt-4o-mini"
    #     assert tree.settings.BASE_PROVIDER == "openai"

    #     # change the complex model
    #     configure(complex_model="gpt-4o", complex_provider="openai")

    #     # should now be changed (no error)
    #     complex_lm_loaded_in_tree = tree.get_lm("complex")

    #     assert isinstance(complex_lm_loaded_in_tree, dspy.LM)
    #     assert complex_lm_loaded_in_tree.model == "openai/gpt-4o"
    #     assert tree.settings.COMPLEX_MODEL == "gpt-4o"
    #     assert tree.settings.COMPLEX_PROVIDER == "openai"

    def test_change_llm_in_tree_local(self):
        """
        When using local settings, global settings should NOT be reflected in the tree
        """
        settings = Settings()

        # create a Tree with local settings obj
        tree = Tree(settings=settings, low_memory=True)

        # should raise an error, no models set
        with pytest.raises(IncorrectModelError):
            tree.base_lm

        with pytest.raises(IncorrectModelError):
            tree.complex_lm

        # change the models by global configure
        configure(
            base_model="gpt-4o-mini",
            base_provider="openai",
            complex_model="gpt-4o",
            complex_provider="openai",
        )

        # should still error because the tree doesnt use global configure
        with pytest.raises(IncorrectModelError):
            tree.base_lm

        with pytest.raises(IncorrectModelError):
            tree.complex_lm

        # change the models by local configure
        settings.configure(
            base_model="gpt-4o-mini",
            base_provider="openai",
            complex_model="gpt-4o",
            complex_provider="openai",
        )

        # should now be no errors
        base_lm_loaded_in_tree = tree.base_lm
        assert isinstance(base_lm_loaded_in_tree, dspy.LM)
        assert base_lm_loaded_in_tree.model == "openai/gpt-4o-mini"
        assert tree.settings.BASE_MODEL == "gpt-4o-mini"
        assert tree.settings.BASE_PROVIDER == "openai"

        complex_lm_loaded_in_tree = tree.complex_lm
        assert isinstance(complex_lm_loaded_in_tree, dspy.LM)
        assert complex_lm_loaded_in_tree.model == "openai/gpt-4o"
        assert tree.settings.COMPLEX_MODEL == "gpt-4o"
        assert tree.settings.COMPLEX_PROVIDER == "openai"

        self.reset_global_settings()

    def test_change_llm_with_debug(self):
        """
        When using debug, the models should be loaded into the Tree and not changed
        """
        settings = Settings()

        # create a Tree with local settings obj, should not error
        tree = Tree(settings=settings, low_memory=False)

        # change the models by global configure
        settings.configure(
            local=True,
            base_model="gpt-4o-mini",
            base_provider="openai",
            complex_model="gpt-4o",
            complex_provider="openai",
        )

        # should now be changed (no error)
        tree = Tree(settings=settings, low_memory=False)

        base_lm_loaded_in_tree = tree.base_lm
        assert isinstance(base_lm_loaded_in_tree, dspy.LM)
        assert base_lm_loaded_in_tree.model == "openai/gpt-4o-mini"
        assert tree.settings.BASE_MODEL == "gpt-4o-mini"
        assert tree.settings.BASE_PROVIDER == "openai"

        complex_lm_loaded_in_tree = tree.complex_lm
        assert isinstance(complex_lm_loaded_in_tree, dspy.LM)
        assert complex_lm_loaded_in_tree.model == "openai/gpt-4o"
        assert tree.settings.COMPLEX_MODEL == "gpt-4o"
        assert tree.settings.COMPLEX_PROVIDER == "openai"

        self.reset_global_settings()
