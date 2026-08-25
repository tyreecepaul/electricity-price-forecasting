import os
from datetime import datetime

import nemosis
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemosis_cache")
TABLE = "DISPATCHPRICE"

DEFAULT_REGIONS = ["NSW1"]
DEFAULT_START = datetime(2024, 1, 1)
DEFAULT_END = datetime(2024, 1, 31)
DEFAULT_CLEAN = True

TEST_FRACTION = 0.2


class Data:
    """Shared data model passed between pipeline stages."""

    def __init__(self, df: pd.DataFrame, regions: list[str], start: datetime, end: datetime):
        self.df = df
        self.regions = regions
        self.start = start
        self.end = end


def _ingest(regions: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)

    start_str = start.strftime("%Y/%m/%d %H:%M:%S")
    end_str = end.strftime("%Y/%m/%d %H:%M:%S")

    df = nemosis.dynamic_data_compiler(
        start_str,
        end_str,
        TABLE,
        CACHE_DIR,
        filter_cols=["REGIONID"],
        filter_values=(regions,),
    )
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["INTERVENTION"] == 0]
    df = df[["SETTLEMENTDATE", "REGIONID", "RRP"]].dropna(subset=["RRP"])
    df = df.rename(
        columns={
            "SETTLEMENTDATE": "settlement_date",
            "REGIONID": "region",
            "RRP": "rrp",
        }
    )
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["region"] = df["region"].astype(str)
    df["rrp"] = df["rrp"].astype(float)
    df = df.sort_values("settlement_date").reset_index(drop=True)
    return df


def _split(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    test_fraction: float = TEST_FRACTION,
    clean: bool = DEFAULT_CLEAN,
):
    date_col = "settlement_date" if clean else "SETTLEMENTDATE"
    cutoff = start + (end - start) * (1 - test_fraction)
    train_df = df[df[date_col] < cutoff].reset_index(drop=True)
    test_df = df[df[date_col] >= cutoff].reset_index(drop=True)
    return train_df, test_df


def getTrainingData(
    regions: list[str] = DEFAULT_REGIONS,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    clean: bool = DEFAULT_CLEAN,
) -> Data:
    df = _ingest(regions, start, end)
    if clean:
        df = _clean(df)
    train_df, _ = _split(df, start, end, clean=clean)
    return Data(train_df, regions, start, end)


def getTestData(
    regions: list[str] = DEFAULT_REGIONS,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    clean: bool = DEFAULT_CLEAN,
) -> Data:
    df = _ingest(regions, start, end)
    if clean:
        df = _clean(df)
    _, test_df = _split(df, start, end, clean=clean)
    return Data(test_df, regions, start, end)
