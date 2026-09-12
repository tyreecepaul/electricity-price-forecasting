from abc import ABC, abstractmethod
from data_cleaning.data import Data
import numpy as np

class Model(ABC):

    model_name: str 

    @abstractmethod 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, training_data: Data) -> None:
        pass

    @abstractmethod
    def predict(self, test_data: Data) -> np.ndarray:
        pass

    def get_model_name(self) -> str:
        return self.model_name