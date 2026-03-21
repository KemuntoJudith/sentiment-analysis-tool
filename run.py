# Starts the backend server. 
# It initializes the Flask app and registers the API routes defined in the `app.api.routes` module. 
# The server runs in debug mode for development purposes.

from flask import Flask
from app.api.routes import api_routes
from app.api.analytics_routes import analytics_routes

app = Flask(__name__)

# Register prediction API
app.register_blueprint(api_routes)

# Register analytics API
app.register_blueprint(analytics_routes, url_prefix="/analytics")

if __name__ == "__main__":
    app.run(debug=True)#