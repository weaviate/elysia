from elysia.tree.objects import Objects

class GenericAggregation(Objects):
    """
    Objects:
        List of dictionaries.
        Each one has keys:
            - "property_name": name of the field in the collection
            - "aggregate_type": type of aggregation that was performed
            - "aggregate_value": value of the aggregation

    """
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
