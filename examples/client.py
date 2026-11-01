"""A small client for `sepal serve`: one alert, two questions.

    python examples/client.py [--url http://127.0.0.1:8000] [--api-key KEY]

Standard library only. The body is the TypeSafe `systemone` body unchanged: `state` (a string or any JSON value) and `questions`
(id -> {type, instructions, criteria}); `type` is `noul` (yes/no), `choice` or `score`.
"""
import argparse
import json
import urllib.request

BODY = {
    "model": "sepal",
    "state": "rule: Multiple failed logons followed by a success\nuser: svc-backup\nsource_ip: 203.0.113.9\nevent_count: 41\nwindow: 5 minutes",
    "questions": {
        "escalate": {"type": "noul", "instructions": "Should a human analyst look at this alert now?"},
        "verdict": {"type": "choice", "instructions": "What is the most likely cause?",
                    "criteria": {"benign": "expected activity", "false_positive": "the rule fired wrongly", "malicious": "an attack"}},
    },
}


def call(url: str, path: str, body=None, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    print(call(args.url, "/v1/models", key=args.api_key))
    print(json.dumps(call(args.url, "/v1/systemone", BODY, args.api_key), indent=2))
