# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyHermes'
copyright = '2024, PyHermes Team'
author = 'PyHermes Team'
release = 'https://github.com/pyhermes.git'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_copybutton',
    'recommonmark',
    'sphinx_markdown_tables',
    'notfound.extension',
    'versionwarning.extension',
    'nbsphinx',
    'sphinx-prompt',
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