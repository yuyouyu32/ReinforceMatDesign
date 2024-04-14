from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    