"""TeamArtemisSE489.

An ML movie recomender
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("teamartemisse489")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "TeamArtemisIV"
__email__ = "ahern255@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
