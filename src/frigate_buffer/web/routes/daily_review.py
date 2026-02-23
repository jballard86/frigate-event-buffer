"""Daily review blueprint: API for listing and reading daily reports, generate POST."""

from datetime import date, datetime

from flask import Blueprint, jsonify, request

from frigate_buffer.web.report_helpers import get_report_for_date, list_report_dates


def create_bp(orchestrator):
    """Create daily_review API blueprint with routes closed over orchestrator."""
    bp = Blueprint("daily_review", __name__, url_prefix="/api/daily-review")
    storage_path = orchestrator.config["STORAGE_PATH"]

    @bp.route("/dates")
    def daily_review_dates():
        dates = list_report_dates(storage_path)
        return jsonify({"dates": dates})

    @bp.route("/current")
    def daily_review_current():
        today = date.today()
        data = get_report_for_date(storage_path, today)
        if data:
            return jsonify(data)
        return jsonify({"error": "No report for today yet"}), 404

    @bp.route("/<date_str>")
    def daily_review_get(date_str):
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format"}), 400
        data = get_report_for_date(storage_path, d)
        if data:
            return jsonify(data)
        return jsonify({"error": "Report not found for this date"}), 404

    @bp.route("/generate", methods=["POST"])
    def daily_review_generate():
        date_str = (request.args.get("date") or "").strip()
        if not date_str:
            body = request.get_json(silent=True)
            if isinstance(body, dict) and body.get("date"):
                date_str = str(body.get("date", "")).strip()
        if not date_str:
            date_str = date.today().isoformat()
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format"}), 400
        if orchestrator.daily_reporter is None:
            return jsonify({"error": "Daily reporter disabled (AI not enabled)"}), 503
        if orchestrator.daily_reporter.generate_report(d):
            return jsonify({"success": True, "date": d.isoformat()}), 200
        return jsonify({"error": "Report generation failed"}), 503

    return bp
