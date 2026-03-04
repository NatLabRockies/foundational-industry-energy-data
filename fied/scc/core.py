
import os

import pandas as pd
import polars as pl
import numpy as np
import re
import logging
import requests
import yaml
import sys
from pathlib import Path
from io import BytesIO
from unit_matcher import UnitsFuels

from fied import datasets
from fied.scc.unit_matcher import UnitsFuels
from fied.scc.misc import match_fuel_type, char_nei_units, load_scc_fuel_types, map_fuel_types
from fied.scc.industrial_process import id_industrial_processes

from fied.utils import expect_polars, expect_pandas

# Transition solution
_unit_methods = UnitsFuels()


def _classify_space_heater(row: dict) -> dict:
    lv4 = row["scc_level_four"]
    ut = char_nei_units(row["scc_level_two"])
    ft1, ft2 = match_fuel_type(lv4.split(": ")[0] if ":" in lv4 else lv4)
    return {
        "unit_type_lv1": ut["unit_type_lv1"],
        "unit_type_lv2": ut["unit_type_lv2"],
        "fuel_type_lv1": ft1,
        "fuel_type_lv2": ft2,
    }


def _classify_boiler(row: dict) -> dict:
    lv3 = row["scc_level_three"]
    lv4 = row["scc_level_four"]
    ut1 = "Boiler"
    ut_match = re.search(r"(?<=Boiler,\s)[\w\s\W]+|(?<=Coal:\s)[\w\s\W]+", lv4)

    if ut_match:
        matched = ut_match.group()
        if "Boiler, " in matched:
            ut2 = matched.split("Boiler, ")[1]
        elif "Pulverizd Coal:" in matched:
            ut2 = matched.split(": ")[1]
        else:
            ut2 = matched
        ft1, ft2 = match_fuel_type(lv3)

    else:
        fuel_types = load_scc_fuel_types()
        if lv4 in fuel_types:
            ft1, ft2 = match_fuel_type(lv4)
            ut2 = "Boiler"
        elif lv4 == "All":
            ft1, ft2 = match_fuel_type(lv3)
            ut2 = "Boiler"
        else:
            ft1, ft2 = match_fuel_type(lv3)
            ut2 = lv4

    return {
        "unit_type_lv1": ut1,
        "unit_type_lv2": ut2,
        "fuel_type_lv1": ft1,
        "fuel_type_lv2": ft2,
    }


_EXTERNAL_COMBUSTION_DTYPE = pl.Struct({
    "unit_type_lv1": pl.String,
    "unit_type_lv2": pl.String,
    "fuel_type_lv1": pl.String,
    "fuel_type_lv2": pl.String,
})


def id_external_combustion(scc: pl.LazyFrame) -> pl.LazyFrame:
    """
    Method for identifying relevant unit and fuel types under
    SCC Level 1 External Combustion (1)

    Temporary solution during transition to polars.

    Parameters
    ----------
    scc : pl.LazyFrame
        Complete list of SCCs.

    Returns
    -------
    pl.LazyFrame
        SCC for external combustion (SCC Level 1 == 1) with
        unit type and fuel type descriptions.
    """
    scc_exc = scc.filter(pl.col("scc_level_one") == "External Combustion")

    space_heaters = (
        scc_exc
        .filter(pl.col("scc_level_two") == "Space Heaters")
        .with_columns(
            pl.struct("scc_level_two", "scc_level_four")
            .map_elements(_classify_space_heater, return_dtype=_EXTERNAL_COMBUSTION_DTYPE)
            .alias("_types")
        )
        .unnest("_types")
    )

    boilers = (
        scc_exc
        .filter(pl.col("scc_level_two").str.contains("Boilers"))
        .with_columns(
            pl.struct("scc_level_three", "scc_level_four")
            .map_elements(_classify_boiler, return_dtype=_EXTERNAL_COMBUSTION_DTYPE)
            .alias("_types")
        )
        .unnest("_types")
    )

    return pl.concat([space_heaters, boilers])


@expect_pandas
def old_id_external_combustion(all_scc):
    """
    Method for identifying relevant unit and fuel types under
    SCC Level 1 External Combustion (1)

    Parameters
    ----------
    all_scc : pandas.DataFrame
        Complete list of SCCs.

    Returns
    -------
    scc_exc : pandas.DataFrame
        SCC for external combustion (SCC Level 1 == 1) with
        unit type and fuel type descriptions.
    """

    scc_exc = all_scc.query("scc_level_one=='External Combustion'")

    all_types = {
        'unit_type_lv1': [],
        'unit_type_lv2': [],
        'fuel_type_lv1': [],
        'fuel_type_lv2': []
        }

    for i, r in scc_exc.iterrows():

        if r['scc_level_two'] == 'Space Heaters':
            ut = char_nei_units(r['scc_level_two'])
            ut1 = ut["unit_type_lv1"]
            ut2 = ut["unit_type_lv2"]

            if ':' in r['scc_level_four']:
                ft1, ft2 = match_fuel_type(r['scc_level_four'].split(': ')[0])

            else:
                ft1, ft2 = match_fuel_type(r['scc_level_four'])

        elif 'Boilers' in r['scc_level_two']:

            ut1 = "Boiler"

            ut_match = re.search(r'(?<=Boiler,\s)[\w\s\W]+|(?<=Coal:\s)[\w\s\W]+', r['scc_level_four'])

            if ut_match:

                if ((':' in ut_match.group()) & ('Boiler, ' in ut_match.group())) | ('Boiler, ' in ut_match.group()):

                    ut2 = ut_match.group().split('Boiler, ')[1]

                elif 'Pulverizd Coal:' in ut_match.group():

                    ut2 = ut_match().split(': ')[1]

                else:

                    ut2 = ut_match.group()

                ft1, ft2 = match_fuel_type(r['scc_level_three'])

            else:

                if r['scc_level_four'] in (load_scc_fuel_types().keys()):

                    ft1, ft2 = match_fuel_type(r['scc_level_four'])
                    ut2 = 'Boiler'

                elif r['scc_level_four'] == 'All':

                    ft1, ft2 = match_fuel_type(r['scc_level_three'])
                    ut2 = 'Boiler'

                else:
                    ft1, ft2 = match_fuel_type(r['scc_level_three'])

                    ut2 = r['scc_level_four']

        all_types['unit_type_lv1'].append(ut1)
        all_types['unit_type_lv2'].append(ut2)
        all_types['fuel_type_lv1'].append(ft1)
        all_types['fuel_type_lv2'].append(ft2)

    scc_exc = scc_exc.join(
        pd.DataFrame(all_types, index=scc_exc.index)
        )

    return scc_exc


def id_stationary_fuel_combustion(all_scc: pl.LazyFrame) -> pl.LazyFrame:
    """Add 2-tier unit and fuel for SCC Level 1 == 2

    Filter the given SCC for "stationary fuel combustion", i.e. SCC
    Level 1 equal to 2, and for those, define unit and fuel in a
    2-tier levels.

    Parameters
    ----------
    all_scc : pl.LazyFrame
        Complete list of SCCs.

    Returns
    -------
    pl.LazyFrame
        Filtered SCC rows with added columns: ``unit_type_lv1``,
        ``unit_type_lv2``, ``fuel_type_lv1``, ``fuel_type_lv2``.
    """

    lv4 = pl.col("scc_level_four")

    return (
        all_scc
        .filter(
            (pl.col("scc_level_one") == "Stationary Source Fuel Combustion")
            & (pl.col("scc_level_two") != "Residential")
        )
        # -- Unit types: determined by scc_level_four ----
        .with_columns(
            pl.when(lv4.str.contains("All Boiler Types"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Boiler"),
                unit_type_lv2=pl.lit("Boiler"),
            ))
            .when(lv4.str.contains("Boilers and IC Engines"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Other combustion"),
                unit_type_lv2=pl.lit("Boilers and internal combustion engines"),
            ))
            .when(lv4.str.contains("All IC Engine Types"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Internal combustion engine"),
                unit_type_lv2=pl.lit("Internal combustion engine"),
            ))
            .when(lv4.str.contains("All Heater Types"))
            .then(pl.struct(
                unit_type_lv1=pl.lit("Heater"),
                unit_type_lv2=pl.lit("Heater"),
            ))
            .otherwise(pl.struct(
                unit_type_lv1=pl.lit("Other combustion"),
                unit_type_lv2=lv4,
            ))
            .alias("_unit_types")
        )
        .unnest("_unit_types")
        # -- Fuel types -----------------------------------
        .pipe(map_fuel_types, fuel_type_col="scc_level_three")
    ).collect().to_pandas().set_index("index")


@expect_polars
def id_chemical_evaporation(scc: pl.LazyFrame) -> pl.LazyFrame:
    """Identify relevant unit and fuel types under SCC Level 1
    Chemical Evaporation (4)

    Parameters
    ----------
    scc : pl.LazyFrame
        Complete list of SCCs.

    Returns
    -------
    pl.LazyFrame
        Filtered SCC rows for Chemical Evaporation (SCC Level 1 == 4)
        with added columns: ``unit_type_lv1``, ``unit_type_lv2``,
        ``fuel_type_lv1``, ``fuel_type_lv2``.
    """

    _LV2 = pl.col("scc_level_two")
    _LV3 = pl.col("scc_level_three")
    _LV4 = pl.col("scc_level_four")

    # ── Condition expressions (evaluated in priority order) ──────
    is_dryer = _LV4.str.to_lowercase().str.contains("dryer|drying")
    is_oven_gen = _LV3 == "Coating Oven - General"
    has_angle_bracket = _LV4.str.contains(r"[<>]")
    is_oven_heater = _LV3 == "Coating Oven Heater"
    is_ffe_surface = (_LV3 == "Fuel Fired Equipment") & (_LV2 == "Surface Coating Operations")
    is_ffe_organic = (_LV3 == "Fuel Fired Equipment") & (_LV2 == "Organic Solvent Evaporation")

    return (
        scc
        .filter(
            (pl.col("scc_level_one") == "Chemical Evaporation")
            & _LV2.is_in([
                "Printing/Publishing",
                "Surface Coating Operations",
                "Organic Solvent Evaporation",
            ])
        )
        # -- Unit types ----
        .with_columns(
            pl.when(is_dryer)
            .then(pl.struct(
                unit_type_lv1=pl.lit("Dryer"),
                unit_type_lv2=_LV4,
            ))
            .when(is_oven_gen & has_angle_bracket)
            .then(pl.struct(
                unit_type_lv1=pl.lit("Oven"),
                unit_type_lv2=pl.lit("Coating Oven"),
            ))
            .when(is_oven_gen)
            .then(pl.struct(
                unit_type_lv1=pl.lit("Oven"),
                unit_type_lv2=_LV4,
            ))
            .when(is_oven_heater)
            .then(pl.struct(
                unit_type_lv1=pl.lit("Heater"),
                unit_type_lv2=pl.lit("Coating oven heater"),
            ))
            .when(is_ffe_surface)
            .then(pl.struct(
                unit_type_lv1=pl.lit("Other"),
                unit_type_lv2=_LV4.str.split(": ").list.last(),
            ))
            .when(is_ffe_organic)
            .then(pl.struct(
                unit_type_lv1=pl.lit("Other combustion"),
                unit_type_lv2=_LV4.str.split(": ").list.first(),
            ))
            .alias("_unit_types"),
        )
        .unnest("_unit_types")
        # -- Fuel types ----
        .with_columns(
            pl.when(is_oven_heater)
            .then(_LV4)
            .when(is_ffe_surface)
            .then(_LV4.str.split(": ").list.first())
            .when(is_ffe_organic)
            .then(_LV4.str.split(": ").list.last())
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("_fuel_key"),
        )
        .pipe(map_fuel_types, fuel_type_col="_fuel_key")
        # .unnest("_fuel_types")
        # Null out fuel for rows where no fuel key was derived
        # Should this logic be in map_fuel_types?
        .with_columns(
            pl.when(pl.col("_fuel_key").is_not_null())
            .then(pl.col("fuel_type_lv1"))
            .alias("fuel_type_lv1"),
            pl.when(pl.col("_fuel_key").is_not_null())
            .then(pl.col("fuel_type_lv2"))
            .alias("fuel_type_lv2"),
        )
        .drop("_fuel_key")
        .filter(
            pl.col("unit_type_lv1").is_not_null()
            | pl.col("unit_type_lv1").is_not_null()
        )
    ).collect().to_pandas().set_index("index")


# Known ICE sub-type keywords in scc_level_four
_ICE_TYPE_KEYWORDS = ["Turbine", "Reciprocating", "2-cycle", "4-cycle",
        # 'Turbine: Cogeneration',
        # 'Reciprocating: Cogeneration',
        # 'Refinery Gas: Turbine',
        # 'Refinery Gas: Reciprocating Engine',
        # 'Propane: Reciprocating',
        # 'Butane: Reciprocating',
        # 'Reciprocating Engine',
        # 'Reciprocating Engine: Cogeneration'
                      ]

# scc_level_three values that are not combustion-related and should be excluded
_EXCLUDED_LV3 = [
    "Geysers/Geothermal",
    "Equipment Leaks",
    "Wastewater, Aggregate",
    "Wastewater, Points of Generation",
    "Flares",
]


@expect_polars
def id_internal_combustion_engine(all_scc: pl.LazyFrame) -> pl.LazyFrame:
    """Determine unit and fuel types for Internal Combustion Engines

    SCC Level 1 => Internal Combustion Engines (2)

    Parameters
    ----------
    all_scc : pl.LazyFrame
        Complete list of SCCs.

    Returns
    -------
    pl.LazyFrame
        Filtered SCC rows with added columns: ``unit_type_lv1``,
        ``unit_type_lv2``, ``fuel_type_lv1``, ``fuel_type_lv2``.
    """
    # Regex pattern that matches any of the ICE type keywords
    types_pattern = "|".join(_ICE_TYPE_KEYWORDS)

    # Set of known fuel-type keys for the membership check
    known_fuel_keys = list(load_scc_fuel_types().keys())

    lv3 = pl.col("scc_level_three")
    lv4 = pl.col("scc_level_four")

    # Boolean expressions reused across unit and fuel logic
    has_type_kw = lv4.str.contains(types_pattern)
    lv4_is_fuel = lv4.is_in(known_fuel_keys)

    return (
        all_scc
        .filter(
            (pl.col("scc_level_one") == "Internal Combustion Engines")
            & pl.col("scc_level_two").is_in(
                ["Electric Generation", "Industrial",
                 "Commercial/Institutional", "Engine Testing"]
            )
            & ~lv3.is_in(_EXCLUDED_LV3)
        )
        # -- Unit types ----
        .with_columns(
            pl.lit("Internal combustion engine")
            .alias("unit_type_lv1")
        )
        .with_columns(
            pl.when(has_type_kw)
            .then(lv4)
            .otherwise(
                lv3.str.split("Testing").list.first()
            ).alias("unit_type_lv2")
        )
        # -- Fuel types: derive a single lookup key, then join ----
        .with_columns(
            pl.when(has_type_kw)
            .then(lv3)
            .when(lv4_is_fuel)
            .then(lv4)
            .otherwise(pl.lit("Jet A Fuel"))
            .alias("_fuel_key")
        )
        .pipe(map_fuel_types, fuel_type_col="_fuel_key")
        .drop("_fuel_key")
    ).collect().to_pandas().set_index("index")


class SCC_ID:
    """
    Use descriptions of SCC code levels to identify unit type and fuel type 
    indicated by a complete SCC code (e.g., 30190003). 
    The U.S. EPA uses Source Classification Codes (SCCs) to 
    classify different types of activities that generate emissions. 
    Each SCC represents a unique source category-specific process or
    function that emits air pollutants. The SCCs are used as a primary
    identifying data element in EPA’s WebFIRE (where SCCs are
    used to link emissions factors to an emission process),
    the National Emissions Inventory (NEI), and other EPA databases.

    Eight digit SCC codes, such as ABBCCCDD, are structured as follows:

    A: Level 1
    BB: Level 2
    CCC: Level 3
    DD: Level 4

    See SCC documentation for additional information:
    https://sor-scc-api.epa.gov/sccwebservices/sccsearch/docs/SCC-IntroToSCCs_2021.pdf

    """

    def __init__(self):

        logging.basicConfig(level=logging.INFO)

        self._unit_methods = UnitsFuels()

        self._FIEDPATH = Path(__file__).parents[1]

        self._complete_scc_filepath = Path(self._FIEDPATH, "data/SCC/SCCDownload.csv")

        self._all_fuel_types_path = Path(self._FIEDPATH, "tools/all_fuels.csv")

        # YAML that contains fuel types
        self._all_fuel_types_path = Path(self._FIEDPATH, "scc/fuel_type_standardization.yaml")


        # self._all_fuel_types = pd.read_csv(self._all_fuel_types_path)
        # self._all_fuel_types = pd.read_csv(self._all_fuel_types_path, index_col=['ft'])
        # self._all_fuel_types = self._all_fuel_types[~self._all_fuel_types.index.duplicated()]  # Catch duplicates
        # self._all_fuel_types = self._all_fuel_types.to_dict(orient='index')

        with open(self._all_fuel_types_path, 'r') as file:
            self._all_fuel_types = yaml.safe_load(file)


    @staticmethod
    def scc_query_split(scc):
        """
        Uses EPA SCC Web Services to get level information for an 8- or
        10-digit SCC.

        Parameters
        ----------
        scc : int
            Eight or 10-digit Source Classification Code.

        Returns
        -------
        scc_levels : dict
            Dictionary of all four level names
        """

        base_url = 'http://sor-scc-api.epa.gov:80/sccwebservices/v1/SCC/'

        try:
            r = requests.get(base_url+f'{scc}')
            r.raise_for_status()

        except requests.exceptions.HTTPError as err:
            logging.error(f'{err}')

        levels = [f'scc level {n}' for n in ['one', 'two', 'three', 'four']]

        scc_levels = {}

        try:
            for k in levels:
                scc_levels[k] = r.json()['attributes'][k]['text']

        except TypeError:
            logging.error(f'SCC {scc} cannot be found')
            scc_levels = None

        return scc_levels

    def build_id(self):
        """
        Identify all relevant unit types and fuel
        types in SCCs. Expanded to include multi-level unit types
        and fuel types. The following tables list the first level unit and
        fuel types. The fuel types are based in part on `EPA GHGRP Table C-1 to Subpart C <https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-98/subpart-C/appendix-Table%20C-1%20to%20Subpart%20C%20of%20Part%2098>`_
        Unit types are . Note that not all unit types of interest are combustion units. 

        .. csv-table:: Level 1 Unit Types
            :header: "Unit Type"
    
            "Boiler"
            "Furnace"
            "Heater"
            "Dryer"
            "Kiln"
            "Internal combustion engine"'
            "Oven"
            "Combined cycle"
            "Turbine"
            "Other combustion"
            "Other"
        
        .. csv-table:: Level 1 Fuel Types
            :header: "Fuel Type"
    
            "Coal and coke"
            "Natural gas"
            "Petroleum products"
            "Biomass"
            "Other"

        Returns
        -------
        all_scc : pandas.DataFrame
            Complete SCC codes with added columns
            of unit types and fuel types.
        """

        all_scc =  datasets.fetch_scc().collect().to_pandas()
        id_meth = [
             id_external_combustion,
             id_stationary_fuel_combustion,
             id_ice,
             id_chemical_evaporation,
             id_industrial_processes,
            ]

        ids = pd.concat(
            [f(all_scc) for f in id_meth],
            axis=0, ignore_index=False,
            sort=True
            )
        
        # TODO fix where rows with NaN SCC and lists for fuel types and unit types are being generated.
        # This is a stopgap fix
        ids.dropna(subset=['SCC'], inplace=True)

        all_scc = all_scc.join(
            ids[['unit_type_lv1', 'unit_type_lv2', 'fuel_type_lv1','fuel_type_lv2']]
            )

        # all_scc.dropna(subset=['unit_type_lv1', 'unit_type_lv2', 'fuel_type_lv1',
        #                        'fuel_type_lv2'],
        #                how='all', inplace=True)

        return all_scc


    def main(self):
        id_scc_df = self.build_id()
        os.makedirs(self._FIEDPATH / 'scc', exist_ok=True)
        id_scc_df.to_csv(self._FIEDPATH / 'scc' / 'iden_scc.csv')


if __name__ == '__main__':

    id_scc = SCC_ID().main()
