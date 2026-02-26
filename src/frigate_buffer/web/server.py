"""Flask app factory: creates app and registers blueprints."""

import os

from flask import Flask

from frigate_buffer.web.routes import (
    create_api_bp,
    create_daily_review_bp,
    create_pages_bp,
    create_proxy_bp,
    create_test_bp,
)


def create_app(orchestrator):
    """Create Flask app with all endpoints.

    Registers blueprints; routes close over orchestrator.
    """
    _template_dir = os.path.join(os.path.dirname(__file__), "templates")
    _static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, template_folder=_template_dir, static_folder=_static_dir)

    @app.before_request
    def _count_request():
        with orchestrator._request_count_lock:
            orchestrator._request_count += 1

    app.register_blueprint(create_pages_bp(orchestrator))
    app.register_blueprint(create_proxy_bp(orchestrator))
    app.register_blueprint(create_daily_review_bp(orchestrator))
    app.register_blueprint(create_api_bp(orchestrator))
    app.register_blueprint(create_test_bp(orchestrator))

    return app
