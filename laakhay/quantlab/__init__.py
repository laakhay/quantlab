"""
Quantlab - Quantitative analysis tools for the Laakhay ecosystem
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Laakhay Corporation"

def hello():
    print("Hello from Laakhay Quantlab!")

__all__ = ['hello']
