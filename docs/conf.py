# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
from pathlib import Path


def _get_version():
    init = Path(__file__).parent.parent / "primus" / "__init__.py"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init.read_text(), re.MULTILINE)
    if not match:
        raise ValueError("Could not find __version__ in primus/__init__.py")
    return match.group(1)


# Project info
version = _get_version()
release = version
project = f"AMD Primus {version}"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."

# Theme-related configs
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "ai-ecosystem",
    "link_main_doc": True,
    "repository_url": "https://github.com/AMD-AGI/Primus",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}
html_title = project

# Sphinx extension-related configs
extensions = ["rocm_docs"]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "primus"

# Publish the llms.txt index at the docs site root and let
# rocm-docs-core generate llms-full.txt after each build (the llms.txt standard,
# https://llmstxt.org/). See the rocm-docs-core guide:
# https://rocm.docs.amd.com/projects/rocm-docs-core/en/latest/user_guide/llms.html
rocm_docs_generate_llms = True
