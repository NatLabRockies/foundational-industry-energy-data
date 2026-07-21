# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'The Foundational Industry Energy Dataset (FIED)'
copyright = '2026, Alliance for Energy Innovation, LLC'
author = 'NLR: Colin McMillan and Carrie Schoeneberger'

release = get_version("fied").split("+")[0]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages', # Adds .nojekyll, useful for branch deploy
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = [
    "**/.ipynb_checkpoints/",
    "**/__pycache__/**",
    "**/includes/**",
    "**/build/**",
    "**/.DS_Store/**",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Move to pydata_sphinx_theme
html_theme = 'press'
html_static_path = ['_static']

# -- Options for LaTeX / PDF output ------------------------------------------
# Use XeLaTeX with an explicit fontspec-based font setup. This avoids the
# default pdflatex font stack (cmap / CM-Super / T1 encodings), which pulls
# in optional TeX Live packages that are not always installed and have been
# breaking the PDF build. DejaVu ships in texlive-fonts-recommended, which
# is already installed in the CI environment.
latex_engine = "xelatex"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
    # Replace Sphinx's default font package block with an explicit
    # fontspec configuration so we do not depend on cmap/fontenc/CM-Super.
    "fontpkg": r"""
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
""",
    # Skip Sphinx's cmap package inclusion (pdflatex-only, not needed with
    # XeLaTeX and its Unicode-native font handling).
    "cmappkg": "",
    # fontenc is unnecessary under XeLaTeX; fontspec handles encoding.
    "fontenc": "",
    # inputenc is a no-op under XeLaTeX (input is already Unicode).
    "inputenc": "",
}

# -- Extension configuration -------------------------------------------------

# -- Autodoc & Autosummary configuration --
add_module_names = False  # Remove namespaces from class/method signatures
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
autodoc_typehints = "none"
autosummary_generate = True                   # Auto-generate stub pages
autosummary_generate_overwrite = True         # Regenerate stubs on every build
autosummary_imported_members = False          # Skip re-exported names

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource", # Keep methods in source-code order
}

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
# linkcheck_anchors_ignore
# linkcheck_ignore

# -- Napoleon configuration --
napoleon_google_docstring = False
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
napoleon_type_aliases = {
    "pd.DataFrame": "pandas.DataFrame",
    "pd.Series": "pandas.Series",
    "np.ndarray": "numpy.ndarray",
    "np.array": "numpy.ndarray",
    "pl.DataFrame": "polars.DataFrame",
    "pl.LazyFrame": "polars.LazyFrame",
}

#napoleon_attr_annotations = True

# -- Suppress cross-reference warnings for unresolvable types --
nitpick_ignore_regex = [
    (r"py:class", r"optional"),              # NumPy docstring convention ",     optional"
]

# Standalone analysis scripts that execute file I/O or use Py2-style
# imports at module top level; exclude from recursive autosummary.
_AUTOSUMMARY_SKIP_MODULES = {
    "ghgrp_unit_analysis",
    "nei_emissions_calc_methods",
    "nei_industrial_sector",
    "nei_unit_analysis",
    "scc_describe",
    "onsite_food",
    "food_qpc",
}


def _patch_autosummary_module_scan():
    # Recursive autosummary discovers submodules via pkgutil.iter_modules
    # in sphinx.ext.autosummary.generate._get_modules, bypassing the
    # autodoc-skip-member event. Wrap it to drop blocklisted short names.
    from sphinx.ext.autosummary import generate as _gen

    _original = _gen._get_modules

    def _filtered(obj, **kwargs):
        public, all_ = _original(obj, **kwargs)
        public = [m for m in public if m.rsplit(".", 1)[-1] not in _AUTOSUMMARY_SKIP_MODULES]
        all_ = [m for m in all_ if m.rsplit(".", 1)[-1] not in _AUTOSUMMARY_SKIP_MODULES]
        return public, all_

    _gen._get_modules = _filtered


_patch_autosummary_module_scan()


def setup(app):
    pass
