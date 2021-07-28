"""
pySHOC Photometry pipeline runner. 
"""

# This just runs the `main` function from `pyshoc.pipeline.main` in the event 
# that the pipeline is invoked via
# >>> python shoc/pipeline /path/to/data


if __name__ != '__main__':
    raise SystemExit()


import sys
from shoc.pipeline.main import main


main(*sys.argv[1:])




# ---------------------------------------------------------------------------- #

#
# chronos.mark('Logging setup')
# ---------------------------------------------------------------------------- #