# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "..").resolve()))
sys.path.insert(0, str(Path("..", "..", "examples", "convergence").resolve()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SKHiPPR"
copyright = "2025, Fabia Bayer"
author = "Fabia Bayer"
release = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
