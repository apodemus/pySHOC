# !/usr/bin/env python

"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""

# This will print the welcome banner and run the `main` function from
# `pyshoc.pipeline.main` in the event that the pipeline is invoked via
# >>> python pyshoc /path/to/data

if __name__ == '__main__':
    # This will print the banner
    from pyshoc.pipeline.run import main

    # run the pipeline
    main()
