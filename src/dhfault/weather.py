"""Real hourly outdoor temperature used to drive the simulator.

The bundled file ``data/zurich_2022_2023_temp.csv`` is 2 m air temperature for Zurich, 2022-01-01 to
2023-12-31, hourly, downloaded from the Open-Meteo historical archive
(https://open-meteo.com/en/docs/historical-weather-api, data licence CC BY 4.0, "Weather data by
Open-Meteo.com"). The archive is a reanalysis product (ERA5/ERA5-Land based), not a MeteoSwiss
station series. Times are local (Europe/Zurich) wall-clock times without offset.
"""

from __future__ import annotations

from importlib import resources

import pandas as pd


def load_hourly_temperature() -> pd.DataFrame:
    """Return a DataFrame indexed by hourly timestamps with column ``temperature_2m`` (deg C)."""
    with resources.files("dhfault").joinpath("data/zurich_2022_2023_temp.csv").open("rb") as fh:
        df = pd.read_csv(fh, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps in bundled weather file")
    return df
