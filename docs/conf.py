# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.util import find_spec

import pyhermes

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyHermes'
copyright = '2026, PyHermes Team'
author = 'PyHermes Team'
release = pyhermes.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

_optional_extensions = [
    'sphinx_copybutton',
    'recommonmark',
    'sphinx_markdown_tables',
    'notfound.extension',
    'versionwarning.extension',
    'nbsphinx',
    'sphinx_prompt',
]


def _extension_available(name):
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        base_name = name.split('.')[0]
        return find_spec(base_name) is not None


extensions = ['sphinx.ext.mathjax'] + [
    ext for ext in _optional_extensions if _extension_available(ext)
]

# Sphinx extensions:
# https://sphinx-extensions.readthedocs.io/en/latest/

templates_path = ['_templates']
exclude_patterns = []

# favicons = [
#    {
#       "sizes": "16x16",
#       "href": "https://secure.example.com/favicon/favicon-16x16.png",
#    },
#    {
#       "sizes": "32x32",
#       "href": "https://secure.example.com/favicon/favicon-32x32.png",
#    },
#    {
#       "rel": "apple-touch-icon",
#       "sizes": "180x180",
#       "href": "apple-touch-icon-180x180.png",  # use a local file in _static
#    },
# ]

# Sphinx user guide:
# https://sublime-and-sphinx-guide.readthedocs.io/

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# Some other themes, see:
# https://nbsphinx.readthedocs.io/en/0.9.5/usage.html#3rd-Party-Themes

html_static_path = ['_static']
html_css_files = ['style.css']
# html_show_sourcelink = False

# html_logo = ''
# html_favicon = ''
