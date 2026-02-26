"""
Gemini proxy HTTP client - POST to OpenAI-compatible /v1/chat/completions with retry.

Used by GeminiAnalysisService for all proxy requests. Encapsulates URL, headers,
tuning params, and 2-attempt retry (second attempt: Accept-Encoding: identity,
Connection: close).
"""

import json
import logging

import requests
import urllib3.exceptions

from frigate_buffer.constants import LOG_MAX_RESPONSE_BODY

logger = logging.getLogger("frigate-buffer")


def _log_proxy_failure(proxy_url: str, attempt: int, exc: Exception) -> None:
    """Log proxy request failure with URL, optional response body,
    and a hint for connection errors."""
    err_name = type(exc).__name__
    err_msg = str(exc)
    hint = ""
    if "refused" in err_msg.lower() or "Connection refused" in err_msg:
        hint = f" Ensure the AI proxy is running at {proxy_url}."
    response_body = ""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            err_obj = data.get("error") if isinstance(data, dict) else None
            if isinstance(err_obj, dict) and err_obj.get("message"):
                response_body = json.dumps(
                    {
                        "error": {
                            "message": err_obj.get("message"),
                            "type": err_obj.get("type", "unknown"),
                        }
                    }
                )
            else:
                response_body = json.dumps(data) if data is not None else ""
        except (ValueError, TypeError, AttributeError):
            try:
                text = getattr(resp, "text", None) or (
                    resp.content.decode("utf-8", errors="replace")
                    if getattr(resp, "content", None)
                    else ""
                )
                response_body = (text or "")[:LOG_MAX_RESPONSE_BODY]
            except Exception:
                response_body = repr(getattr(resp, "content", None))[
                    :LOG_MAX_RESPONSE_BODY
                ]
        if len(response_body) > LOG_MAX_RESPONSE_BODY:
            response_body = response_body[:LOG_MAX_RESPONSE_BODY] + "..."
    if response_body:
        logger.error(
            "Proxy request failed [%s]. URL: %s. Body: %s. attempt=%s/2.%s",
            getattr(resp, "status_code", "?"),
            proxy_url,
            response_body,
            attempt,
            hint,
        )
    else:
        logger.error(
            "Proxy request failed on attempt %s/2: %s: %s. url=%s.%s",
            attempt,
            err_name,
            err_msg,
            proxy_url,
            hint,
        )


class GeminiProxyClient:
    """Gemini proxy client (OpenAI-compatible). POST with retry; caller parses."""

    def __init__(
        self,
        proxy_url: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        self._proxy_url = proxy_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

    def post_messages(self, messages: list, timeout: int) -> requests.Response | None:
        """
        POST messages to /v1/chat/completions with retry (second attempt: identity
        encoding, Connection: close).
        Returns the response on success, None on failure. Caller parses response body.
        """
        url = f"{self._proxy_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
        }
        base_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(2):
            headers = dict(base_headers)
            if attempt == 1:
                headers["Accept-Encoding"] = "identity"
                headers["Connection"] = "close"
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=timeout
                )
                resp.raise_for_status()
                return resp
            except (
                requests.exceptions.ChunkedEncodingError,
                urllib3.exceptions.ProtocolError,
            ) as e:
                logger.warning(
                    "Proxy attempt %s/2 failed with ChunkedEncodingError: %s. "
                    "Retrying with 'Connection: close'...",
                    attempt + 1,
                    e,
                )
                continue
            except Exception as e:
                _log_proxy_failure(url, attempt + 1, e)
                if attempt == 1:
                    return None
                continue
        return None
