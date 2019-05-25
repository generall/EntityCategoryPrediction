import os

from category_prediction.service import app

host = os.getenv('HOST', '127.0.0.1')

app.run(port=5000, host=host)

# To Run:
# python run.py
# or
# python -m flask run
