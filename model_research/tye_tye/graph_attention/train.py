"""Train and evaluate the spatial-only GAT model across all 5 NEM regions.

Note: evaluation/evaluation.py's evaluate() is currently a non-functional
stub (imports "Model" instead of "evaluation.Model", and assumes a
test_data.rrp attribute that doesn't exist on the Data class), so this script
computes MAE/RMSE per region directly instead of calling it.
"""

from __future__ import annotations

import numpy as np
/
from gat_model import GATModel
from graph_data import REGION_ORDER, load_graph_data

HORIZON_MINUTES = 30
TARGET_TYPE = "change"
SEQUENCE_LENGTH = 288


def main() -> None:
    train_batch, test_batch = load_graph_data(
        regions=REGION_ORDER,
        horizon=HORIZON_MINUTES,
        target=TARGET_TYPE,
        sequence_length=SEQUENCE_LENGTH,
    )

    n_features = len(train_batch.feature_columns)
    model = GATModel(n_features=n_features, epochs=20)

    print(
        f"train samples={train_batch.X.shape[0]} "
        f"test samples={test_batch.X.shape[0]} "
        f"features={n_features} nodes={len(REGION_ORDER)}"
    )

    model.train(train_batch)
    predictions = model.predict(test_batch)  # [n_samples, num_nodes]

    print(f"\nFINAL TEST RESULTS ({model.get_model_name()}, target={TARGET_TYPE})")
    for node_index, region in enumerate(REGION_ORDER):
        y_true = test_batch.y[:, node_index]
        y_pred = predictions[:, node_index]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        print(f"  {region}: MAE={mae:.4f} RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
