"""Sphinx configuration for the DYANA documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from dyana.version import __version__

project = "DYANA"
copyright = "2026, Hiro"
author = "Hiro"

release = os.environ.get("DYANA_DOCS_VERSION", __version__)
version = ".".join(release.split(".")[:3])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"
master_doc = "index"
default_role = "literal"

html_theme = "pydata_sphinx_theme"
html_title = "DYANA"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_last_updated_fmt = "%b %d, %Y"
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "show_prev_next": False,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "secondary_sidebar_items": ["page-toc"],
    "article_header_start": ["breadcrumbs"],
    "icon_links": [
        {
            "name": "Repository",
            "url": "https://github.com/CogSci-hiro/dyana",
            "icon": "fa-brands fa-github",
        }
    ],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "default_mode": "light",
}

autosummary_generate = True
autosummary_imported_members = False
autodoc_mock_imports = [
    "dyana.evidence.overlap",
    "dyana.eval.suite",
    "soundfile",
    "webrtcvad",
    "praatio",
]
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_custom_sections = [
    ("Usage example", "Examples"),
    ("Design Notes", "Notes"),
    ("Rules", "Notes"),
    ("Behavior", "Notes"),
    ("Formula", "Notes"),
]
