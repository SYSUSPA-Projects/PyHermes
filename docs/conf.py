# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from importlib.util import find_spec
from pathlib import Path

import pypandoc
import pyhermes

# ``pypandoc-binary`` keeps notebook rendering self-contained on CI and local
# docs builds. nbconvert invokes ``pandoc`` through PATH.
_pandoc_dir = str(Path(pypandoc.get_pandoc_path()).parent)
os.environ['PATH'] = _pandoc_dir + os.pathsep + os.environ.get('PATH', '')

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
    'sphinx_prompt',
]


def _extension_available(name):
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        base_name = name.split('.')[0]
        return find_spec(base_name) is not None


extensions = [
    'sphinx.ext.mathjax',
    'nbsphinx',
    'nbsphinx_link',
] + [
    ext for ext in _optional_extensions if _extension_available(ext)
]

# Render the outputs committed in examples/notebooks without rerunning
# scientific calculations during a documentation build.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False

# Sphinx extensions:
# https://sphinx-extensions.readthedocs.io/en/latest/

templates_path = ['_templates']
exclude_patterns = ['_build', '**/.ipynb_checkpoints']

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

html_theme = 'pydata_sphinx_theme'
html_title = 'PyHermes'

html_static_path = ['_static']
html_css_files = ['style.css']
# html_show_sourcelink = False

html_favicon = '_static/pyhermes_mark_round.png'

html_theme_options = {
    'logo': {
        'text': 'PyHermes',
        'alt_text': 'PyHermes documentation - Home',
        'image_light': '_static/pyhermes_mark_round.png',
        'image_dark': '_static/pyhermes_mark_round.png',
    },
    'navbar_align': 'left',
    'navbar_center': [],
    'navigation_with_keys': True,
    'show_nav_level': 2,
    'show_toc_level': 2,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/SYSUSPA-Projects/PyHermes',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/pyhermes-cosmo/',
            'icon': 'fa-solid fa-box',
        },
    ],
}

html_sidebars = {
    '**': ['pyhermes-sidebar-logo.html'],
}
