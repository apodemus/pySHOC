"""
Create a snapshot of sutherland weather page for nightlog
"""


# std
import io
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

# third-party
from PIL import Image


URL = 'http://suthweather.saao.ac.za/image.png'


def get_png(path):
    """
    Retrieve and save png image of Sutherland environmental monitoring page
    """

    response = urllib.request.urlopen(URL)
    stream = io.BytesIO(response.read())

    img = Image.open(stream)
    t = datetime.now()
    # morning hours --> backdate image to start day of observing run
    t -= timedelta(0 < t.hour < 12)

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    date = str(t.date()).replace('-', '')
    filename = path / f'env{date}.png'
    # TODO log print(filename)
    img.save(filename)


if __name__ == '__main__':

    import argparse
    from sys import argv

    parser = argparse.ArgumentParser(
        prog='weather.py',
        description='Grabs a png image of the sutherland weather status page'
    )
    # Positional arguments
    parser.add_argument(
        'path',  
        help='Folder location for resulting image'
    )

    args = parser.parse_args(argv[1:])

    # get weather page snapshot
    get_png(args.path)

# def create():
    # run this after your observing night to generate a log of the observations
    # as a
