"""
Global Flask Application Setting

See `.flaskenv` for default settings.
 """

import os

APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(APP_DIR)
DIST_DIR = os.path.join(ROOT_DIR, 'service', 'dist')
