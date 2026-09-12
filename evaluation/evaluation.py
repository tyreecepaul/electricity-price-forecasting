import numpy as np
from preprocessing import PreparedModelData
from evaluation.Model import Model

def evaluate(model: Model, test_data: PreparedModelData) -> None:
    y_true: np.ndarray = test_data.test_df["target_rrp"].to_numpy()
    y_pred: np.ndarray = model.predict(test_data)

    squared_losses: np.ndarray = (y_true - y_pred) ** 2
    sum_of_squared_losses: float = float(np.sum(squared_losses))

    print(f"SSE for {model.get_model_name()}: {sum_of_squared_losses:.4f}")
