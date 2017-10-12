from __future__ import absolute_import
import os


__version__ = '0.1.0'

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(*path):
    return os.path.join(_ROOT, 'data', *path)
