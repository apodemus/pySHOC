# !/usr/bin/env python

"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""

# This just runs the `main` function from `pyshoc.pipeline.main` in the event
# that the pipeline is invoked via
# >>> python shoc /path/to/data


if __name__ == '__main__':
    from pyshoc import CONFIG
    
    #
    if CONFIG.console.banner.show:
        from pyshoc.pipeline import WELCOME_BANNER


        # say hello
        print(WELCOME_BANNER)

    #
    from pyshoc.pipeline.main import main

    main.main()
