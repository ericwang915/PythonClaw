#!/usr/bin/env python3
"""GitHub API client.  Reads token from pythonclaw.json."""

import argparse
import json
import os
import re
import sys

try:
    import requests
except ImportError:
    print("Error: requests not installed.  Run: pip install requests", file=sys.stderr)
    sys.exit(1)

API = "https://api.github.com"


def _load_token() -> str:
    for path in ["pythonclaw.json", os.path.expanduser("~/.pythonclaw/pythonclaw.json")]:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        data = json.loads(text)
        token = data.get("skills", {}).get("github", {}).get("token", "")
        if token:
            return token
    return ""


def _headers() -> dict:
    token = _load_token()
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(path: str) -> dict | list:
    resp = requests.get(f"{API}{path}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{API}{path}", headers=_headers(), json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def cmd_repos(user: str) -> None:
    repos = _get(f"/users/{user}/repos?sort=updated&per_page=30")
    for r in repos:
        stars = r.get("stargazers_count", 0)
        lang = r.get("language", "")
        desc = r.get("description", "") or ""
        print(f"  {r['full_name']}  [{lang}]  \u2605{stars}")
        if desc:
            print(f"    {desc[:100]}")


def cmd_repo(full_name: str) -> None:
    r = _get(f"/repos/{full_name}")
    print(json.dumps({
        "name": r["full_name"],
        "description": r.get("description", ""),
        "language": r.get("language"),
        "stars": r["stargazers_count"],
        "forks": r["forks_count"],
        "open_issues": r["open_issues_count"],
        "url": r["html_url"],
        "created": r["created_at"],
        "updated": r["updated_at"],
    }, indent=2))


def cmd_issues(full_name: str) -> None:
    issues = _get(f"/repos/{full_name}/issues?state=open&per_page=30")
    for i in issues:
        if i.get("pull_request"):
            continue
        labels = ", ".join(lb["name"] for lb in i.get("labels", []))
        print(f"  #{i['number']}  {i['title']}")
        if labels:
            print(f"    Labels: {labels}")


def cmd_create_issue(full_name: str, title: str, body: str) -> None:
    result = _post(f"/repos/{full_name}/issues", {"title": title, "body": body})
    print(f"Created issue #{result['number']}: {result['html_url']}")


def cmd_prs(full_name: str) -> None:
    prs = _get(f"/repos/{full_name}/pulls?state=open&per_page=30")
    for pr in prs:
        print(f"  #{pr['number']}  {pr['title']}  ({pr['user']['login']})")


def cmd_pr(full_name: str, number: int) -> None:
    pr = _get(f"/repos/{full_name}/pulls/{number}")
    print(json.dumps({
        "number": pr["number"],
        "title": pr["title"],
        "state": pr["state"],
        "author": pr["user"]["login"],
        "branch": pr["head"]["ref"],
        "base": pr["base"]["ref"],
        "body": (pr.get("body") or "")[:500],
        "url": pr["html_url"],
        "mergeable": pr.get("mergeable"),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="GitHub API client.")
    sub = parser.add_subparsers(dest="command")

    p_repos = sub.add_parser("repos")
    p_repos.add_argument("user")

    p_repo = sub.add_parser("repo")
    p_repo.add_argument("full_name", help="owner/repo")

    p_issues = sub.add_parser("issues")
    p_issues.add_argument("full_name", help="owner/repo")

    p_ci = sub.add_parser("create-issue")
    p_ci.add_argument("full_name", help="owner/repo")
    p_ci.add_argument("--title", required=True)
    p_ci.add_argument("--body", default="")

    p_prs = sub.add_parser("prs")
    p_prs.add_argument("full_name", help="owner/repo")

    p_pr = sub.add_parser("pr")
    p_pr.add_argument("full_name", help="owner/repo")
    p_pr.add_argument("number", type=int)

    args = parser.parse_args()

    try:
        if args.command == "repos":
            cmd_repos(args.user)
        elif args.command == "repo":
            cmd_repo(args.full_name)
        elif args.command == "issues":
            cmd_issues(args.full_name)
        elif args.command == "create-issue":
            cmd_create_issue(args.full_name, args.title, args.body)
        elif args.command == "prs":
            cmd_prs(args.full_name)
        elif args.command == "pr":
            cmd_pr(args.full_name, args.number)
        else:
            parser.print_help()
    except requests.HTTPError as exc:
        print(f"GitHub API error: {exc.response.status_code} {exc.response.text[:200]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
