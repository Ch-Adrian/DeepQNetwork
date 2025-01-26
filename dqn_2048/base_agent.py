from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, discount) -> None:
        self.discount = discount

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def act(self):
        pass
