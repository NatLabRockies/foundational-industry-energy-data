
from importlib.resources import files
import yaml


def load_scc_fuel_types():
    """Load the fuel types"""
    yaml_path = files('fied.scc').joinpath('fuel_type_standardization.yaml')
    with yaml_path.open('r') as f:
        return yaml.safe_load(f)
