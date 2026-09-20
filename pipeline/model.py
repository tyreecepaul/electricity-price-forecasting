from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Model(ABC):

    model_name: str 

    @abstractmethod 
    def __init__(self, **model_params) -> None:
        pass

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        pass

    def get_model_name(self) -> str:
        return self.model_name