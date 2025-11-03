import sys
import json


def die(exit_code: int, msg: str, detail: str = None, extra: dict | None = None):
    payload = {"error": msg}

    if detail:
        payload["detail"] = detail

    if extra:
        payload["extra"] = extra

    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.exit(exit_code)


def require(condition: bool, msg: str, *, exit_code: int, detail: str = None):
    if not condition:
        die(exit_code, msg, detail)
