from __future__ import annotations

import pytest

from research_assistant.web import WebError, _check_public_host, normalize_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8080/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://[::1]/",
        "http://10.0.0.5/internal",
    ],
)
def test_check_public_host_blocks_private_and_loopback(url):
    with pytest.raises(WebError):
        _check_public_host(url)


def test_check_public_host_blocks_non_http_scheme():
    with pytest.raises(WebError):
        _check_public_host("file:///etc/passwd")


def test_normalize_url_lowercases_and_strips_fragment():
    assert normalize_url("HTTPS://Example.COM/Path?q=1#frag") == "https://example.com/Path?q=1"
    assert normalize_url("https://example.com") == "https://example.com/"
