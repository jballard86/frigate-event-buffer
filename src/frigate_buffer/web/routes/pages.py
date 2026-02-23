"""Pages blueprint: player, stats-page, daily-review, test-multi-cam."""

from flask import Blueprint, redirect, render_template, request


def create_bp(orchestrator):
    """Create pages blueprint with routes closed over orchestrator."""
    bp = Blueprint("pages", __name__)

    @bp.route("/player")
    def player():
        if request.args.get("filter") == "stats":
            return redirect("/stats-page")
        return render_template(
            "player.html",
            stats_refresh_seconds=orchestrator.config.get("STATS_REFRESH_SECONDS", 60),
        )

    @bp.route("/stats-page")
    def stats_page():
        return render_template(
            "stats.html",
            stats_refresh_seconds=orchestrator.config.get("STATS_REFRESH_SECONDS", 60),
        )

    @bp.route("/daily-review")
    def daily_review_page():
        return render_template("daily_review.html")

    @bp.route("/test-multi-cam")
    def test_multi_cam_page():
        subdir = request.args.get("subdir", "")
        return render_template("test_run.html", subdir=subdir)

    return bp
