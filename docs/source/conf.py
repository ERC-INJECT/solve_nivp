# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'solve_nivp'
copyright = '2025, David Riley'
author = 'David Riley'
release = 'March 26, 2025'

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

# Generate autosummary pages automatically
autosummary_generate = True

# Type hints in description for cleaner signatures
autodoc_typehints = 'description'

# Napoleon options
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Intersphinx mappings for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', {}),
    'numpy': ('https://numpy.org/doc/stable/', {}),
    'scipy': ('https://docs.scipy.org/doc/scipy/', {}),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

latex_engine = 'xelatex'
# latex_elements = {
#     'preamble': r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}',
# }

