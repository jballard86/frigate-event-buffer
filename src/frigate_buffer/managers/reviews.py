"""Fetches and stores Frigate daily review summaries."""

import os
import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import requests

logger = logging.getLogger('frigate-buffer')


class DailyReviewManager:
    """Fetches and stores Frigate daily review summaries (POST /api/review/summarize/start/{start}/end/{end})."""

    def __init__(self, storage_path: str, frigate_url: str, retention_days: int):
        self.reviews_dir = os.path.join(storage_path, 'daily_reviews')
        self.frigate_url = frigate_url.rstrip('/')
        self.retention_days = retention_days
        os.makedirs(self.reviews_dir, exist_ok=True)
        logger.info(f"DailyReviewManager: {self.reviews_dir}, retention={retention_days} days")

    def _date_to_ts_range(self, d: date, end_now: bool = False) -> tuple:
        """Return (start_ts, end_ts) for a date. end_now=True uses current time for end."""
        start_dt = datetime.combine(d, datetime.min.time())
        end_dt = datetime.now() if end_now else datetime.combine(d, datetime.max.time().replace(microsecond=0))
        return (int(start_dt.timestamp()), int(end_dt.timestamp()))

    def _date_str(self, d: date) -> str:
        return d.strftime('%Y-%m-%d')

    def _path_for_date(self, d: date) -> str:
        return os.path.join(self.reviews_dir, f"{self._date_str(d)}.json")

    def fetch_from_frigate(self, start_ts: int, end_ts: int) -> Optional[dict]:
        """Fetch review summary from Frigate API."""
        url = f"{self.frigate_url}/api/review/summarize/start/{start_ts}/end/{end_ts}"
        logger.info(f"Fetching daily review from Frigate: {url}")
        try:
            resp = requests.post(url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if data.get('success') and 'summary' in data:
                return data
            logger.warning(f"Frigate returned success=false or no summary: {data}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching daily review from Frigate")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching daily review: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error fetching daily review: {e}")
            return None

    def fetch_and_save(self, d: date, end_now: bool = False) -> Optional[dict]:
        """Fetch from Frigate and save to disk. Returns the response dict or None."""
        start_ts, end_ts = self._date_to_ts_range(d, end_now=end_now)
        data = self.fetch_from_frigate(start_ts, end_ts)
        if data:
            data['start_ts'] = start_ts
            data['end_ts'] = end_ts
            data['date'] = self._date_str(d)
            data['end_now'] = end_now
            path = self._path_for_date(d)
            if end_now:
                path = path.replace('.json', '_partial.json')
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved daily review to {path}")
                return data
            except Exception as e:
                logger.error(f"Failed to save daily review: {e}")
        return None

    def get_cached(self, d: date, allow_partial: bool = False) -> Optional[dict]:
        """Get cached review for date. allow_partial also checks _partial.json for today."""
        path = self._path_for_date(d)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cached review: {e}")
        if allow_partial:
            partial_path = path.replace('.json', '_partial.json')
            if os.path.exists(partial_path):
                try:
                    with open(partial_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error reading partial review: {e}")
        return None

    def get_or_fetch(self, d: date, force_refresh: bool = False, end_now: bool = False) -> Optional[dict]:
        """Get cached review or fetch from Frigate. end_now only applies when force_refresh."""
        if not force_refresh:
            cached = self.get_cached(d, allow_partial=(d == date.today()))
            if cached:
                return cached
        return self.fetch_and_save(d, end_now=end_now)

    def list_dates(self) -> List[str]:
        """Return sorted list of available date strings (YYYY-MM-DD), newest first."""
        dates = set()
        for f in os.listdir(self.reviews_dir):
            if not f.endswith('.json'):
                continue
            date_str = f.replace('_partial.json', '').replace('.json', '')
            if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                dates.add(date_str)
        return sorted(dates, reverse=True)

    def cleanup_old(self) -> int:
        """Remove reviews older than retention. Returns count deleted."""
        cutoff = date.today() - timedelta(days=self.retention_days)
        deleted = 0
        for f in os.listdir(self.reviews_dir):
            if not f.endswith('.json'):
                continue
            date_str = f.replace('.json', '').replace('_partial', '')
            if len(date_str) != 10:
                continue
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
                if d < cutoff:
                    path = os.path.join(self.reviews_dir, f)
                    os.remove(path)
                    deleted += 1
                    logger.info(f"Cleaned up old daily review: {f}")
            except ValueError:
                pass
        return deleted
