#!/usr/bin/env python3
"""Deploy web/ to the linked Vercel project via REST API."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1] / "web"
AUTH = pathlib.Path.home() / ".local/share/com.vercel.cli/auth.json"
PROJECT_ID = "prj_KFCsdydUEWKNPQkSC2JpEVcizeVl"
TEAM_ID = "team_rvUzRUMzifi5tjrt00sY41QM"
PRODUCTION_ALIAS = "bioacoustic-embedding-dynamics.vercel.app"
SKIP_DIRS = {".vercel", ".git"}
SKIP_FILES = {".gitignore"}


def load_token() -> str:
    token = os.environ.get("VERCEL_TOKEN")
    if token:
        return token
    data = json.loads(AUTH.read_text())
    return data["token"]


def iter_files() -> list[tuple[str, pathlib.Path]]:
    out: list[tuple[str, pathlib.Path]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name in SKIP_FILES:
            continue
        out.append((rel.as_posix(), path))
    return out


def sha1_file(path: pathlib.Path) -> tuple[str, int, bytes]:
    data = path.read_bytes()
    digest = hashlib.sha1(data).hexdigest()
    return digest, len(data), data


def api_request(token: str, method: str, url: str, body: bytes | None = None, headers: dict | None = None):
    req_headers = {"Authorization": f"Bearer {token}"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, method=method, headers=req_headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def upload_file(token: str, rel: str, digest: str, size: int, data: bytes) -> None:
    url = f"https://api.vercel.com/v2/files?teamId={TEAM_ID}"
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
        "x-now-digest": digest,
        "x-now-size": str(size),
    }
    try:
        api_request(token, "POST", url, body=data, headers=headers)
    except urllib.error.HTTPError as err:
        if err.code == 409:
            return
        raise


def wait_ready(token: str, deployment_id: str, attempts: int = 40) -> dict:
    import time

    url = f"https://api.vercel.com/v13/deployments/{deployment_id}?teamId={TEAM_ID}"
    last = {}
    for _ in range(attempts):
        last = api_request(token, "GET", url)
        if last.get("readyState") == "READY":
            return last
        if last.get("readyState") in {"ERROR", "CANCELED"}:
            raise RuntimeError(f"deployment failed: {last.get('readyState')}")
        time.sleep(3)
    raise RuntimeError(f"deployment not ready: {last.get('readyState')}")


def assign_alias(token: str, deployment_id: str) -> dict:
    url = f"https://api.vercel.com/v2/deployments/{deployment_id}/aliases?teamId={TEAM_ID}"
    body = json.dumps({"alias": PRODUCTION_ALIAS}).encode()
    headers = {"Content-Type": "application/json"}
    return api_request(token, "POST", url, body=body, headers=headers)


def create_deployment(token: str, files: list[dict]) -> dict:
    url = f"https://api.vercel.com/v13/deployments?teamId={TEAM_ID}"
    payload = {
        "name": "web",
        "files": files,
        "project": PROJECT_ID,
        "target": "production",
        "projectSettings": {
            "framework": None,
            "buildCommand": None,
            "outputDirectory": None,
            "installCommand": None,
        },
        "meta": {
            "githubCommitOrg": "ST-48-1240162",
            "githubCommitRepo": "bioacoustic-embedding-dynamics",
            "githubCommitRef": "main",
        },
    }
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    return api_request(token, "POST", url, body=body, headers=headers)


def main() -> int:
    token = load_token()
    manifest: list[dict] = []
    for rel, path in iter_files():
        digest, size, data = sha1_file(path)
        print(f"upload {rel} ({size} bytes)")
        upload_file(token, rel, digest, size, data)
        manifest.append({"file": rel, "sha": digest, "size": size})
    print(f"creating deployment ({len(manifest)} files)...")
    dep = create_deployment(token, manifest)
    dep_id = dep.get("id")
    if not dep_id:
        raise RuntimeError(f"missing deployment id: {dep}")
    dep = wait_ready(token, dep_id)
    assign_alias(token, dep_id)
    print(
        json.dumps(
            {
                "id": dep_id,
                "url": dep.get("url"),
                "readyState": dep.get("readyState"),
                "alias": PRODUCTION_ALIAS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", "replace")
        print(body, file=sys.stderr)
        raise SystemExit(1)
