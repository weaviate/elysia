import dspy

class RoutingPrompt(dspy.Signature):
    """
    Given a query (user_prompt) from a user, decide a property.
    """

    user_prompt = dspy.InputField(description="The query that the user is asking")
    property_description = dspy.InputField(description="A description of the property that the user is asking about")
    property_options = dspy.InputField(description="The properties that you can choose from")
    property = dspy.OutputField(description="The decided property. This must be one of the options in property_options.")

