
from functools import lru_cache

from importlib.resources import files
import logging
import re
import yaml


def load_scc_fuel_types():
    """Load the fuel types"""
    yaml_path = files('fied.scc').joinpath('fuel_type_standardization.yaml')
    with yaml_path.open('r') as f:
        return yaml.safe_load(f)


def match_fuel_type(ft):
    """
    Match fuel type to an entry in all fuel types yaml. Returns
    standardized level 1 and level 2 fuel types.

    Parameters
    ----------
    ft : str
        Fuel type

    Returns
    -------
    ft1, ft2 : str
        Standardized level 1 and level 2 fuel types.

    Raises
    ------
    KeyError
        If the fuel type is not included in the all fuel types yaml.
    """

    try:
        ft1 = load_scc_fuel_types()[ft]['fuelTypeLv1']
        ft2 = load_scc_fuel_types()[ft]['fuelTypeLv2']

    except KeyError as e:
        logging.error(f"{e}, fuel type: {ft}")

        return "Other", ft

    else:
        return ft1, ft2


@lru_cache(maxsize=1)
def fuel_type_table() -> pl.LazyFrame:
    """Create a lookup LazyFrame from fuel types dictionary (cached)."""
    lookup_data = []
    for ft_key, ft_value in load_scc_fuel_types().items():
        try:
            lookup_data.append({
                "fuel_type": ft_key,
                "fuelTypeLv1": ft_value['fuelTypeLv1'],
                "fuelTypeLv2": ft_value['fuelTypeLv2']
            })
        except (KeyError, TypeError) as e:
            logging.warning(f"Skipping fuel type '{ft_key}': {e}")

    return pl.DataFrame(lookup_data).lazy()


def map_fuel_types(df: pl.LazyFrame, fuel_type_col: str) -> pl.LazyFrame:
    """Add standardized two-tier fuel type as new columns

    Map the given `fuel_type_col` column of `df` using a builtin table
    of a two-tier fuel types.

    Parameters
    ----------
    df : pl.LazyFrame
        Input LazyFrame
    fuel_type_col : str
        Name of column containing fuel type strings

    Returns
    -------
    pl.LazyFrame
        LazyFrame with fuelTypeLv1 and fuelTypeLv2 columns added

    Notes
    -----
    - It assumes that the builtin table does not contain any record with
      `fuelTypeLv1` equal to None, thus when applying a left-join, any
      None is resulted from missing match on the reference table (right).
    - It assumes that `fuel_type_table()` reference column is named
    `fuel_type`.
    """

    return (
        df
        .join(
            fuel_type_table(),
            left_on=fuel_type_col,
            right_on="fuel_type",
            how="left"
        )
        .with_columns([
            # Handle missing matches: if null, use defaults
            pl.when(pl.col("fuelTypeLv1").is_null())
            .then(pl.lit("Other"))
            .otherwise(pl.col("fuelTypeLv1"))
            .alias("fuelTypeLv1"),

            pl.when(pl.col("fuelTypeLv2").is_null())
            .then(pl.col(fuel_type_col))
            .otherwise(pl.col("fuelTypeLv2"))
            .alias("fuelTypeLv2")
        ])
    )


def load_scc_unit_types():
    """Load the SCC unit types"""
    yaml_path = files('fied.scc').joinpath('scc_unit_types.yaml')
    with yaml_path.open('r') as f:
        return yaml.safe_load(f)


def char_nei_units(nei_unit):
    """Characterizes a unit type with a standardized level 1 name.

    Parameters
    ----------
    nei_unit : str
        Name of unit type taken from SCC or NEI data.

    Returns
    -------
    unit_types : list
        List of standardized level 1 and level 2
    """
    unit_type_table = load_scc_unit_types()
    matched = [re.search(nei_unit, y) for y in unit_type_table.keys()]
    matched = [x for x in matched if x != None]

    if len(matched) == 1:
        try:
            ut1 = unit_type_table[matched[0].group()]['unitTypeLv1']
            ut2 = unit_type_table[matched[0].group()]['unitTypeLv2']

        except KeyError as e:
            logging.error(f"Type not in _nei_uts: {e}")

    elif len(matched) > 1:
        ut1, ut2 = 'Other combustion', matched[0].group()
    else:
        ut1, ut2 = "Other", nei_unit

    return {"unit_type_lv1": ut1, "unit_type_lv2": ut2}
