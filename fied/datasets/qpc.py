"""Fetch Census Quarterly Survey of Plant Capacity (QPC) data

Downloads quarterly operating hours and utilization rates by NAICS code
from the U.S. Census Bureau.
"""

import logging
import urllib

import pandas as pd
import polars as pl
import pooch


module_logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = [
    "NAICS",
    "Description",
    "Utilization Rate",
    "UR_Standard Error",
    "Weekly_op_hours",
    "Hours_Standard Error",
]

# SHA-256 hashes for known QPC files. Files not listed here are fetched
# without verification (known_hash=None), which is the original behavior.
_KNOWN_HASHES = {
    "2017_qtr_table_final_q1.xlsx": "sha256:9839eb5b32e2722fb3e38f6ad4c29cb678032eb35b44354cf64cfad89e919caa",
    "2017_qtr_table_final_q2.xlsx": "sha256:a06ea334dc2b8d3c1d18e891393a99a1493d1ce313964b7257958c318d63764f",
    "2017_qtr_table_final_q3.xlsx": "sha256:9af153d11e6ae3ee8376e9f081f894dec98e28ae9b6df1c5aef8161ce912dcc6",
    "2017_qtr_table_final_q4.xlsx": "sha256:e2927a491a1e8a5add954583456474ebb87ece524ae9f9009eb504ed0235ab87",
    "2019_qtr_table_final_q1.xlsx": "sha256:f3412ae9d6f831eb9bcd55ae3005448232bdb6791ad03d6fc39582574a1ad70e",
    "2019_qtr_table_final_q2.xlsx": "sha256:e5e8d74dbee8c258e8203d4fdcb6b128aa664d51ed7935580c4170dad9976919",
    "2019_qtr_table_final_q3.xlsx": "sha256:531ad948026713b4a7e000041979cddb81c403f5904113b8aafec78db451230a",
    "2019_qtr_table_final_q4.xlsx": "sha256:ad22ba33893545581061d2677982bf9995492e8cbf850e03cdc760c61e8a81d5",
    "2020-qtr-table-final-q1.xlsx": "sha256:619a351a8ae7c39139bab23b3248e41a3476c9674a4fa39b782f31b49e1af022",
    "2020-qtr-table-final-q2.xlsx": "sha256:3332e6806f6ed8984d71ee3fe2d6c43eeb77c81f4169122f01ed149dd9634ed8",
    "2020-qtr-table-final-q3.xlsx": "sha256:475f44c9646d6f848d79e709865255d6bc6724e89352e0848f58e3e39fad0690",
    "2020_qtr_table_final_q4.xlsx": "sha256:93886980180fdf3ec2e5509b5b7e04108311b3a2e85a5c9626b5dfeec87e46e8",
    "2022-qtr-table-final-q1.xlsx": "sha256:a67a278bdab928227dcc006e0c0c94a0f1555c0f18a915d90ae90d56c542be41",
    "2022-qtr-table-final-q2.xlsx": "sha256:34d50fc8963edb34ade784c918b6d17083ac6a5965c8cf43a23de80a987333eb",
    "2022-qtr-table-final-q3.xlsx": "sha256:fc630bccf8e91e7d53d2feb8e1cc5b63fc9bbdd46c6d95519a4fd5318badd445",
    "2022-qtr-table-final-q4.xlsx": "sha256:ac8bfbb7aa685a8aad0e2433a1a07409eccb946c1c231ac8682e0f36d4767edb",
}


def fetch_QPC(year):
    """Fetch Quarterly Survey of Plant Capacity data for a given year

    Quarterly survey began 2008; start with 2010 due to 2007-2009
    recession.

    Parameters
    ----------
    year : int
        The reporting year to fetch.

    Returns
    -------
    pl.DataFrame
        Combined quarterly data with columns: NAICS, Description,
        Utilization Rate, UR_Standard Error, Weekly_op_hours,
        Hours_Standard Error, Q, Year.
    """
    y = str(year)

    if year < 2017:
        excel_ex = ".xls"
    else:
        excel_ex = ".xlsx"

    quarterly_frames = []

    base_url = "https://www2.census.gov/programs-surveys/qpc/tables/"

    for q in [f"q{n}" for n in range(1, 5)]:
        if (year >= 2017) & (year < 2020):
            filename = "{!s}/{!s}_qtr_table_final_"

        # elif year < 2010:
        #
        #     filename = \
        #         '{!s}/qpc-quarterly-tables/{!s}_qtr_combined_tables_final_'

        elif (year == 2020) & (q == "q4"):
            filename = "{!s}/{!s}_qtr_table_final_"

        elif year > 2019:
            filename = "{!s}/{!s}-qtr-table-final-"

        else:
            filename = "{!s}/qpc-quarterly-tables/{!s}_qtr_table_final_"

        if (year == 2016) & (q == "q4"):
            filename = filename.format(y, y) + q + ".xlsx?#"

        else:
            filename = filename.format(y, y) + q + excel_ex

        known_hash = _KNOWN_HASHES.get(filename)
        fname = pooch.retrieve(
            base_url + filename, known_hash=known_hash, path=pooch.os_cache("FIED"), progressbar=True
        )

        # Excel formatting for 2008 is different than all other years.
        # Will need to revise skiprows and usecols.
        try:
            raw = pd.read_excel(
                fname, sheet_name=1, skiprows=4, usecols=range(0, 7), header=0
            )

        except urllib.error.HTTPError:
            module_logger.error(f"Problem with {url}: {err}")
            raise

        # Drop positional column (index 2) and convert to polars.
        raw = raw.drop(raw.columns[2], axis=1)
        # Remove rows without data, such as the comments in the bottom.
        # todo! This is a weak criteria and vulnerable to errors.
        raw = raw.dropna()

        # These replacements should be done here, when we can verify
        # what does it mean Z, D, and S.
        # Some cases of: Z  Standard error is less than 0.05.
        # Consider replacing by 0.05 instead of None.
        raw = raw.replace({"Z": None})
        # Some cases of: D  Estimate withheld to avoid disclosing data
        #   for individual companies.
        raw = raw.replace({"D": None})
        # Some cases of: S   Estimate does not meet publication standards
        raw = raw.replace({"S": None})
        raw.columns = _OUTPUT_COLUMNS

        data = (
            pl.from_pandas(raw)
            # null introduced above, avoid removing full rows
            # .drop_nulls()
            .with_columns(
                pl.lit(q).alias("Q"),
                pl.lit(year).alias("Year"),
            )
        )

        quarterly_frames.append(data)

    return pl.concat(quarterly_frames)
