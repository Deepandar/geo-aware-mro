from dataclasses import dataclass


@dataclass
class RegisteredModel:

    name: str

    version: str

    stage: str

    owner: str


class GovernanceRegistry:

    def __init__(self):

        self.models = []

    def register(
        self,
        model: RegisteredModel,
    ):

        self.models.append(model)

    def list_models(self):

        return self.models
