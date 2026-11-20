"""
.. include:: ../../README.md

"""

from .models import FENNIX,register_fennix_module,get_modules_documentation, available_fennix_modules
from .utils.device import configure_device

load = FENNIX.load

try:
    from .version import __version__
except ImportError:  # source tree without a build: no generated version.py
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("FeNNol")
    except PackageNotFoundError:
        __version__ = "unknown"
