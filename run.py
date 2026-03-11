# Starts the Flast backend server. 
# This is the entry point of the application. 
# It initializes the Flask app and registers the API routes defined in the `app.api.routes` module. 
# The server runs in debug mode for development purposes.

from flask import Flask
from app.api.routes import api_routes

app = Flask(__name__)

app.register_blueprint(api_routes)

if __name__ == "__main__":
    app.run(debug=True)