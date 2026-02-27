import unittest
from urllib.parse import urlparse, urlunparse


def mask_url(url):
    try:
        parsed = urlparse(url)
        if parsed.password:
            safe_netloc = f"{parsed.username or ''}:***@{parsed.hostname}"
            if parsed.port:
                safe_netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=safe_netloc)
        return urlunparse(parsed)
    except Exception as e:
        return f"Error masking URL: {e}"


class TestUrlMasking(unittest.TestCase):
    def test_masking_standard(self):
        url = "http://user:password@host:5000"
        expected = "http://user:***@host:5000"
        assert mask_url(url) == expected

    def test_masking_no_port(self):
        url = "http://user:password@host"
        expected = "http://user:***@host"
        assert mask_url(url) == expected

    def test_masking_no_user(self):
        url = "http://:password@host:5000"
        # username is empty string
        expected = "http://:***@host:5000"
        assert mask_url(url) == expected

    def test_no_password(self):
        url = "http://user@host:5000"
        expected = "http://user@host:5000"
        assert mask_url(url) == expected

    def test_basic(self):
        url = "http://host:5000"
        expected = "http://host:5000"
        assert mask_url(url) == expected


if __name__ == "__main__":
    unittest.main()
