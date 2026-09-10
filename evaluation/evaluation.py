import Model
import numpy as np
from data_cleaning import Data

def evaluate(model: Model, test_data: Data) -> None:
    # Extract x and y from test_data
    x: np.ndarray = test_data.rrp
    y_true: np.ndarray = test_data.rrp

    # Make predictions
    y_pred: np.ndarray = model.predict(x)

    # Calculate Sum of Squared Loss
    squared_losses: np.ndarray = np.pow(y_true - y_pred, 2);
    sum_of_squared_losses: float = np.sum(squared_losses);

    print(f"SSE for {model.get_model_name()}: {sum_of_squared_losses}")

