import re
from urllib.parse import urlparse

def parse_url(url: str) -> dict:
    """Parse a URL and return its components.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with keys 'scheme', 'host', and 'path'.

    Raises:
        ValueError: If the URL does not use http or https scheme, or if the host is missing.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    host = parsed.netloc
    if not host:
        raise ValueError("URL must contain a host")
    # Path should be '/' if empty
    path = parsed.path or '/'
    return {
        'scheme': scheme,
        'host': host,
        'path': path,
    }

def normalize_path(path: str | None) -> str:
    """Normalize a URL path.

    - Collapse duplicate '/' characters.
    - Remove trailing '/' unless the path is exactly '/'.
    - Preserve the root path '/'.
    - Accept ``None`` as equivalent to an empty string.
    """
    if not path:
        return '/'  # default to root for empty or None
    # Ensure leading slash exists for consistency
    if not path.startswith('/'):
        path = '/' + path
    # Collapse duplicate slashes
    collapsed = re.sub(r'/+', '/', path)
    # Remove trailing slash if not root
    if collapsed != '/' and collapsed.endswith('/'):
        collapsed = collapsed[:-1]
    return collapsed
