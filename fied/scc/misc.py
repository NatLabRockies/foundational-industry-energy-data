
from importlib.resources import files
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
