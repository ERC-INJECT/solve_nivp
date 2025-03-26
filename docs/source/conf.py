# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'solve_ivp_ns'
copyright = '2025, David Riley'
author = 'David Riley'
release = 'March 26, 2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Optional: if you're using Google or NumPy style docstrings.
    # add any other extensions you need]
    ]

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

