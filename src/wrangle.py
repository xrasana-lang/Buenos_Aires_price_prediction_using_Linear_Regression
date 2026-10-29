import numpy as np
import pandas as pd
from glob import glob


DROP_COLS = [
    "lat-lon", "place_with_parent_names",
    "floor", "expenses", "rooms", "price",
    "price_aprox_local_currency", "price_usd_per_m2",
    "price_per_m2", "operation", "property_type", "currency",
    "properati_url", "surface_total_in_m2",
]


def merged_files(pattern):
    """Read every CSV matching `pattern` and stack into one DataFrame."""
    return pd.concat(
        [pd.read_csv(file) for file in glob(pattern)],
        ignore_index=True,
    )


def filter_df(data):
    return (
        data
        .loc[lambda x: x["place_with_parent_names"].str.contains("Capital Federal", na=False)]
        .query('property_type == "apartment"')
        .query("price_aprox_usd < 400_000")
    )


def outliers_df(data):
    return data.loc[
        lambda x: x["surface_covered_in_m2"].between(
            x["surface_covered_in_m2"].quantile(0.10),
            x["surface_covered_in_m2"].quantile(0.90),
        )
    ]


def modify_cols(data):
    return data.assign(
        lat=lambda x: x["lat-lon"].str.split(",", expand=True)[0].astype(float),
        lon=lambda x: x["lat-lon"].str.split(",", expand=True)[1].astype(float),
        neighborhood=lambda x: x["place_with_parent_names"].str.split("|", expand=True)[3],
    )


def clean_files(pattern):
    return (
        merged_files(pattern)
        .pipe(filter_df)
        .pipe(outliers_df)
        .pipe(modify_cols)
        .drop(columns=DROP_COLS)
        .dropna()
        .reset_index(drop=True)
    )
