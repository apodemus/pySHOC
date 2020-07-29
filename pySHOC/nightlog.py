"""
Generate observing night log
"""


# std libs
import io
from pathlib import Path
from datetime import datetime, timedelta
import urllib

# third-party libs
from PIL import Image


def get_suth_weather_png(path):
    """
    Retrieve and save png image of Sutherland environmental monitoring page
    """
    addy = 'http://suthweather.saao.ac.za/image.png'
    response = urllib.request.urlopen(addy)
    data = response.read()
    stream = io.BytesIO(data)

    img = Image.open(stream)
    t = datetime.now()
    if 0 < t.hour < 12:  # morning hours --> backdate image to start day of observing run
        t -= timedelta(1)

    datestr = str(t.date()).replace('-', '')
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    filename = path / str('env%s.png' % datestr)
    print(filename)
    img.save(filename)


def create():
    # run this after your observing night to generate a log of the observations
    # as a
