from abc import ABC, abstractmethod
import numpy as np
from preprocessing import PreparedModelData

class Model(ABC):

    model_name: str 

    @abstractmethod 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, training_data: PreparedModelData) -> None:
        pass

    @abstractmethod
    def predict(self, test_data: PreparedModelData) -> np.ndarray:
        pass

    def get_model_name(self) -> str:
        return self.model_name