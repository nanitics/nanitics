from importlib.metadata import version as _metadata_version

__version__ = _metadata_version("nanitics")

__all__ = ["__version__"]
