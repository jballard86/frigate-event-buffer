"""Flask blueprints for the web app. Each module exposes create_bp(orchestrator)."""

from frigate_buffer.web.routes.api import create_bp as create_api_bp
from frigate_buffer.web.routes.daily_review import create_bp as create_daily_review_bp
from frigate_buffer.web.routes.pages import create_bp as create_pages_bp
from frigate_buffer.web.routes.proxy_routes import create_bp as create_proxy_bp
from frigate_buffer.web.routes.test_routes import create_bp as create_test_bp

__all__ = [
    "create_api_bp",
    "create_daily_review_bp",
    "create_pages_bp",
    "create_proxy_bp",
    "create_test_bp",
]
