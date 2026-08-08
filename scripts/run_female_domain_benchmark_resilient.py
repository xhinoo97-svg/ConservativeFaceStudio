from __future__ import annotations

import random
import time
import urllib.error
import urllib.request

from app import female_domain_benchmark


_ORIGINAL_URLOPEN = urllib.request.urlopen
_last_request_at = 0.0


def _resilient_urlopen(request, *args, **kwargs):
    global _last_request_at
    minimum_interval = 0.9
    now = time.monotonic()
    delay = minimum_interval - (now - _last_request_at)
    if delay > 0:
        time.sleep(delay)

    last_error = None
    for attempt in range(7):
        try:
            response = _ORIGINAL_URLOPEN(request, *args, **kwargs)
            _last_request_at = time.monotonic()
            return response
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                wait = float(retry_after) if retry_after is not None else 0.0
            except (TypeError, ValueError):
                wait = 0.0
            wait = max(wait, min(30.0, 1.5 * (2 ** attempt)))
            wait += random.uniform(0.15, 0.75)
            time.sleep(wait)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= 3:
                raise
            time.sleep(1.0 + attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("urlopen retry loop exhausted")


def main() -> int:
    urllib.request.urlopen = _resilient_urlopen
    female_domain_benchmark.urllib.request.urlopen = _resilient_urlopen
    return female_domain_benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
