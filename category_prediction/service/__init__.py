import os
from flask import Flask, current_app, send_file

from category_prediction.service.api import api_bp

from .config import *


app = Flask(__name__, static_folder=os.path.join(DIST_DIR, 'static'))
app.register_blueprint(api_bp)

app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'Secret')

app.config['APP_DIR'] = config.APP_DIR
app.config['ROOT_DIR'] = config.ROOT_DIR
app.config['DIST_DIR'] = config.DIST_DIR

app.logger.info('>>> {}'.format(app.config['FLASK_ENV']))


@app.route('/')
def index_client():
    dist_dir = current_app.config['DIST_DIR']
    entry = os.path.join(dist_dir, 'index.html')
    return send_file(entry)


@app.route('/favicon.ico')
def favicon():
    dist_dir = current_app.config['DIST_DIR']
    entry = os.path.join(dist_dir, 'favicon.ico')
    return send_file(entry)
