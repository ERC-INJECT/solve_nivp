# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'solve_nivp'
copyright = '2025, David Riley'
author = 'David Riley'
release = '0.2.0.dev1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # NumPy/Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

# API stubs are committed under docs/source/api. Regenerate them explicitly
# when the public module list changes so normal docs builds do not dirty Git.
autosummary_generate = False

# Type hints in description for cleaner signatures
autodoc_typehints = 'description'

# Optional extras are documented without requiring their runtime dependencies.
autodoc_mock_imports = [
    'gymnasium',
    'stable_baselines3',
    'sb3_contrib',
]

# Napoleon options
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Intersphinx mappings for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = []

latex_engine = 'xelatex'
# latex_elements = {
#     'preamble': r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}',
# }
