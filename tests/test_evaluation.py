from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline.evaluation import (
    ModelResult,
    compare_models,
    print_evaluation_summary,
)


def result(name, predictions, valid_mask=None):
    predictions = np.asarray(predictions, dtype=float)
    if valid_mask is None:
        valid_mask = np.isfinite(predictions)
    return ModelResult(
        model_name=name,
        predictions=predictions,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        metrics={},
    )


def test_compare_models_uses_shared_model_and_aemo_rows():
    test_data = SimpleNamespace(
        horizon=30,
        test_df=pd.DataFrame(
            {
                "target_rrp": [100.0, 200.0, 300.0, 400.0],
                "rrp": [90.0, 210.0, 290.0, 380.0],
                "p5_h30_rrp": [95.0, np.nan, 280.0, 390.0],
            }
        ),
    )
    results = [
        result("Model A", [110.0, 190.0, 310.0, 410.0]),
        result("Model B", [100.0, 200.0, np.nan, 420.0]),
    ]

    comparison = compare_models(results, test_data)

    assert set(comparison.index) == {
        "Model A",
        "Model B",
        "Persistence",
        "AEMO P5MIN",
    }
    assert comparison["n_rows"].eq(2).all()
    assert comparison.loc["Model A", "mae"] == 10.0
    assert comparison.loc["Model B", "mae"] == 10.0
    assert comparison.loc["Persistence", "mae"] == 15.0
    assert comparison.loc["AEMO P5MIN", "mae"] == 7.5


def test_print_evaluation_summary_does_not_dump_predictions(capsys):
    test_data = SimpleNamespace(
        horizon=30,
        test_df=pd.DataFrame(
            {
                "target_rrp": [100.0, 200.0],
                "rrp": [90.0, 190.0],
            }
        ),
    )

    print_evaluation_summary(
        [result("MLP", [101.123456789, 198.0])],
        test_data,
    )

    output = capsys.readouterr().out
    assert "FINAL TEST RESULTS" in output
    assert "Persistence" in output
    assert "101.123456789" not in output
    assert "predictions=array" not in output
