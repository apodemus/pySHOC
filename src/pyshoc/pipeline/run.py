# !/usr/bin/env python

"""
Entry point script for running main pipeline. Print welcome banner before any
heavy duty imports.
"""

from pyshoc.pipeline import WELCOME_BANNER


# say hello
print(WELCOME_BANNER)


def main():
    from pyshoc.pipeline.cli import main
    
    main()
