from elysia.tree.objects import Objects

class Aggregation(Objects):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)

class GenericAggregation(Aggregation):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "aggregation"
