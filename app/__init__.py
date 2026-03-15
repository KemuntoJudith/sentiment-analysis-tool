from flask import Flask
from app.api.routes import api_routes
from app.api.analytics_routes import analytics_routes

def create_app():

    app = Flask(__name__)

    app.register_blueprint(api_routes)
    app.register_blueprint(analytics_routes, url_prefix="/analytics")

    return app

