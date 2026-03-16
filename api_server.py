#  For exposing the API server to the outside world, we use Flask, a lightweight web framework for Python. 
# The API routes are defined in a separate module and registered with the Flask application. 

from flask import Flask
from app.api.routes import api_routes

app = Flask(__name__)

app.register_blueprint(api_routes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)