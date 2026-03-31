# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'The Foundational Industry Energy Dataset (FIED)'
copyright = '2024, Alliance for Sustainable Energy, LLC'
author = 'NREL: Colin McMillan and Carrie Schoeneberger'

release = get_version("fied").split("+")[0]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = [
    "**.ipynb_checkpoints",
    "**__pycache__**",
    "**/includes/**"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# -- Autodoc configuration --
add_module_names = False  # Remove namespaces from class/method signatures
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
autodoc_typehints = "none"

autodoc_type_aliases = {
    "pd.DataFrame": "pandas.DataFrame",
    "pd.Series": "pandas.Series",
    "np.ndarray": "numpy.ndarray",
    "np.array": "numpy.ndarray",
    "pl.DataFrame": "polars.DataFrame",
    "pl.LazyFrame": "polars.LazyFrame",
}

# -- Autosummary configuration --
autosummary_generate = True
autodoc_member_order = "bysource"   # Keep methods in source-code order

# -- BibTeX configuration --
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- Intersphinx configuration --
intersphinx_mapping = {
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "plotly": ("https://plotly.github.io/plotly.py-docs/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}
tls_verify = False

# -- Napoleon configuration --
#napoleon_google_docstring = True
napoleon_numpy_docstring = True
#napoleon_include_init_with_doc = False
#napoleon_include_private_with_doc = False
#napoleon_include_special_with_doc = True
#napoleon_use_admonition_for_examples = False
#napoleon_use_admonition_for_notes = False
#napoleon_use_admonition_for_references = False
#napoleon_use_ivar = False
#napoleon_use_param = True
napoleon_use_rtype = True
#napoleon_preprocess_types = False
#napoleon_type_aliases = None
#napoleon_attr_annotations = True

# -- Suppress cross-reference warnings for unresolvable types --
nitpick_ignore_regex = [
    (r"py:class", r"optional"),              # NumPy docstring convention ",     optional"
]
