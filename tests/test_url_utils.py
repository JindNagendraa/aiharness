import pytest
from src.url_utils import parse_url, normalize_path


def test_parse_https_url():
    url = "https://example.com/users"
    result = parse_url(url)
    assert result == {
        "scheme": "https",
        "host": "example.com",
        "path": "/users",
    }


def test_parse_http_url_without_path():
    url = "http://example.com"
    result = parse_url(url)
    assert result == {
        "scheme": "http",
        "host": "example.com",
        "path": "/",
    }


def test_parse_invalid_scheme():
    with pytest.raises(ValueError):
        parse_url("ftp://example.com")


def test_normalize_duplicate_slashes():
    assert normalize_path("/a//b///c") == "/a/b/c"


def test_normalize_trailing_slash():
    assert normalize_path("/a/b/") == "/a/b"


def test_normalize_root_path():
    assert normalize_path("/") == "/"
    assert normalize_path("") == "/"


def test_normalize_no_leading_slash():
    assert normalize_path("a/b") == "/a/b"
