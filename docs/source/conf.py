
import datetime
import os
import sys

import toml

sys.path.insert(0, os.path.abspath("../.."))


def get_project_data():
    try:
        # Determine the path to pyproject.toml relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pyproject_path = os.path.join(base_dir, "..","..", "pyproject.toml")
        print(f"Reading project meta from {pyproject_path}")
        # Load the pyproject.toml file
        pyproject_data = toml.load(pyproject_path)

        # Extract the version from the project section
        metadata = dict(pyproject_data["project"])
    except Exception as e:
        metadata = "unknown"
    return metadata


metadata = get_project_data()
year = datetime.datetime.now().year

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

if isinstance(metadata, dict):
    project = f"GU Orbit Software {metadata['name']}"
    copyright = f"{year}, {metadata['authors'][0]['name']}"
    author = metadata["authors"][0]["name"]
    release = metadata["version"]
else:
    raise TypeError(
        "metadata must be a dict. There must be a problem with the pyproject.toml file."
    )

def setup(app):
    app.connect("builder-inited", add_jinja_filters)


def add_jinja_filters(app):
    app.builder.templates.environment.filters["extract_last_part"] = extract_last_part


def extract_last_part(fullname):
    return fullname.split(".")[-1]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx_mdinclude",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "docs.source.custom_modules.auto_toctree",
]

autodoc_default_options = {
    "members": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme_options = {
    "dark_css_variables": {
        "color-api-background": "#202020",
        "color-api-background-hover": "#505050",
        "color-sidebar-item-background--current": "#303030",
        "color-sidebar-item-background--hover": "#303030",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "./tf2_py_objects.inv",
    ),
}

autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented_params"

html_static_path = ["style"]
html_css_files = ["custom.css"]