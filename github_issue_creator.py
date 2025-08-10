#!/usr/bin/env python3
"""
GitHub Issue Creator — Parse repo markdown plans and create issues.

Usage examples:
  # Dry-run from repo root using defaults
  python github_issue_creator.py --repo owner/name --dry-run

  # Actually create issues (requires token in env GITHUB_TOKEN)
  GITHUB_TOKEN=ghp_... python github_issue_creator.py --repo owner/name

  # Assign to a user and create milestones per Phase
  GITHUB_TOKEN=... python github_issue_creator.py --repo owner/name --assignee your-username --milestones-per-phase

Notes:
- This script is idempotent by issue title. If an issue with the same title exists (open or closed), it won't create a duplicate unless --force is used.
- It reads these docs by default:
    - development_plan.md
    - data-mining-recommendations.md
- It creates labels if missing (e.g., Phase 2, Priority 1, Docs, Analysis).
- It will skip items marked complete (✓) in development_plan.md by default.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    import requests  # type: ignore
except Exception as e:
    requests = None  # Allow parser-only usage without requests installed


@dataclass
class IssueSpec:
    title: str
    body: str
    labels: List[str] = field(default_factory=list)
    milestone: Optional[str] = None  # milestone title (we'll map to id later)
    assignees: List[str] = field(default_factory=list)


PHASE_HEADING_RE = re.compile(r"^###\s+Phase\s+(\d+)\b.*", re.IGNORECASE)
STATUS_COMPLETE = "✓"
STATUS_IN_PROGRESS = "*"
STATUS_FAILED = "!"


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_sections_by_phase(md_text: str) -> Dict[str, str]:
    """Split development_plan.md into sections keyed by Phase number as string."""
    lines = md_text.splitlines()
    sections: Dict[str, List[str]] = {}
    current_phase: Optional[str] = None
    for line in lines:
        m = PHASE_HEADING_RE.match(line.strip())
        if m:
            current_phase = m.group(1)
            sections[current_phase] = []
            continue
        if current_phase is not None:
            sections[current_phase].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_bullets_with_status(section_text: str) -> List[Tuple[str, Optional[str]]]:
    """Return a list of (bullet_text, status_marker) from a section.
    status_marker is one of '✓', '*', '!', or None if not found.
    We capture bullets at one or two indentation levels starting with '- '.
    """
    results: List[Tuple[str, Optional[str]]] = []
    for line in section_text.splitlines():
        stripped = line.rstrip()
        if re.match(r"^\s{0,4}-\s+", stripped):
            # Extract status markers at end like '... ✓' or '... *' or '... !'
            status = None
            m = re.search(r"(\s[\u2713\*\!])\s*$", stripped)  # ✓ is \u2713
            if m:
                marker = m.group(1).strip()
                if marker in {STATUS_COMPLETE, STATUS_IN_PROGRESS, STATUS_FAILED}:
                    status = marker
                    stripped = stripped[: m.start(1)].rstrip()
            # Normalize whitespace
            bullet_text = re.sub(r"^\s*-\s+", "", stripped)
            results.append((bullet_text, status))
    return results


def build_issue_from_bullet(phase: str, bullet_text: str, file_ref: str, extra_labels: Optional[List[str]] = None) -> IssueSpec:
    title = f"[Phase {phase}] {bullet_text}"
    body = (
        f"Auto-generated from {file_ref} — Phase {phase}.\n\n"
        f"Task: {bullet_text}\n\n"
        f"Please update the description with more context if needed.\n"
        f"Source: {file_ref} (Phase {phase})"
    )
    labels = [f"Phase {phase}"]
    if extra_labels:
        labels.extend(extra_labels)
    return IssueSpec(title=title, body=body, labels=labels)


def parse_development_plan(md_text: str, file_ref: str) -> List[IssueSpec]:
    issues: List[IssueSpec] = []
    sections = extract_sections_by_phase(md_text)
    for phase, section_text in sections.items():
        bullets = parse_bullets_with_status(section_text)
        for bullet_text, status in bullets:
            # Skip obvious meta headings within sections
            if bullet_text.lower().startswith(("status:", "deliverables:", "acceptance criteria:", "risks/", "dependencies:", "notes:")):
                continue
            # Create issues only for non-complete items
            if status == STATUS_COMPLETE:
                continue
            labels_extra: List[str] = []
            # Heuristic labels based on bullet content
            lower = bullet_text.lower()
            if any(k in lower for k in ["doc", "readme", "troubleshooting"]):
                labels_extra.append("Docs")
            if any(k in lower for k in ["test", "unit test", "ci"]):
                labels_extra.append("Testing")
            if any(k in lower for k in ["cli", "argparse", "process_runner"]):
                labels_extra.append("CLI")
            if any(k in lower for k in ["cluster", "topics", "knowledge graph", "code snippet", "similarity", "search", "gap analysis"]):
                labels_extra.append("Analysis")
            issues.append(build_issue_from_bullet(phase, bullet_text, file_ref, labels_extra))

    # Also create issues from "Documentation Updates Needed" section if present
    doc_updates_match = re.search(r"###\s+Documentation Updates Needed.*?(?:(?=\n### )|\Z)", md_text, re.IGNORECASE | re.S)
    if doc_updates_match:
        doc_section = doc_updates_match.group(0)
        for bullet_text, status in parse_bullets_with_status(doc_section):
            if status == STATUS_COMPLETE:
                continue
            spec = IssueSpec(
                title=f"[Docs] {bullet_text}",
                body=f"Auto-generated from {file_ref} — Documentation Updates Needed.\n\nTask: {bullet_text}",
                labels=["Docs"],
            )
            issues.append(spec)

    return issues


def parse_recommendations(md_text: str, file_ref: str) -> List[IssueSpec]:
    issues: List[IssueSpec] = []
    # Map Priority sections to labels
    priority_blocks = re.findall(r"####\s+Priority\s+(\d+)\b.*?(?=(?:\n####\s+Priority\s+\d+\b)|\Z)", md_text, re.IGNORECASE | re.S)
    # The above only returns numbers; re-find blocks with spans
    prio_iter = re.finditer(r"(####\s+Priority\s+(\d+)\b.*?)(?=(?:\n####\s+Priority\s+\d+\b)|\Z)", md_text, re.IGNORECASE | re.S)
    for m in prio_iter:
        block = m.group(1)
        prio_num = m.group(2)
        # Bullets directly under enumerated items (e.g., 1. **Personal Knowledge Graph**)
        # We'll extract the numbered headings as issue titles and include their sub-bullets in the body.
        for item in re.finditer(r"\n\s*\d+\.\s+\*\*(.+?)\*\*\s*(?:\n|$)", block):
            heading = item.group(1).strip()
            # Capture following bullets until next numbered item or end
            start = item.end(0)
            next_item = re.search(r"\n\s*\d+\.\s+\*\*", block[start:])
            sub_block = block[start: start + next_item.start()] if next_item else block[start:]
            sub_bullets = [b for b, _ in parse_bullets_with_status(sub_block)]
            body_lines = [f"Auto-generated from {file_ref} — Priority {prio_num}.", "", f"Initiative: {heading}"]
            if sub_bullets:
                body_lines += ["", "Scope bullets:"] + [f"- {b}" for b in sub_bullets]
            issues.append(
                IssueSpec(
                    title=f"[Priority {prio_num}] {heading}",
                    body="\n".join(body_lines),
                    labels=[f"Priority {prio_num}", "Analysis"],
                )
            )
    return issues


# ---------------- GitHub API helpers ----------------

def gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_repo_api_base(repo: str) -> str:
    return f"https://api.github.com/repos/{repo}"


def ensure_labels(repo: str, token: str, labels: List[str], dry_run: bool) -> None:
    if dry_run:
        return
    url = get_repo_api_base(repo) + "/labels"
    existing = []
    r = requests.get(url, headers=gh_headers(token), params={"per_page": 100})
    if r.ok:
        existing = [l["name"] for l in r.json()]
    for label in labels:
        if label in existing:
            continue
        payload = {"name": label}
        rr = requests.post(url, headers=gh_headers(token), json=payload)
        rr.raise_for_status()


def get_or_create_milestone(repo: str, token: str, title: str, dry_run: bool) -> Optional[int]:
    if dry_run:
        return None
    base = get_repo_api_base(repo)
    r = requests.get(base + "/milestones", headers=gh_headers(token), params={"state": "all", "per_page": 100})
    r.raise_for_status()
    for m in r.json():
        if m.get("title") == title:
            return m.get("number")
    rr = requests.post(base + "/milestones", headers=gh_headers(token), json={"title": title})
    rr.raise_for_status()
    return rr.json().get("number")


def list_existing_issues(repo: str, token: str) -> Dict[str, int]:
    existing: Dict[str, int] = {}
    if token is None or requests is None:
        return existing
    base = get_repo_api_base(repo)
    page = 1
    while True:
        r = requests.get(base + "/issues", headers=gh_headers(token), params={"state": "all", "per_page": 100, "page": page})
        if not r.ok:
            break
        items = r.json()
        if not items:
            break
        for it in items:
            if "pull_request" in it:
                continue  # skip PRs
            existing[it.get("title", "")] = it.get("number", 0)
        page += 1
    return existing


def create_issue(repo: str, token: str, spec: IssueSpec, milestone_numbers: Dict[str, int], dry_run: bool) -> None:
    payload = {
        "title": spec.title,
        "body": spec.body,
        "labels": spec.labels,
    }
    if spec.assignees:
        payload["assignees"] = spec.assignees
    if spec.milestone and spec.milestone in milestone_numbers:
        payload["milestone"] = milestone_numbers[spec.milestone]

    if dry_run:
        print("[DRY-RUN] Would create issue:")
        print(f"  Title: {spec.title}")
        print(f"  Labels: {spec.labels}")
        if spec.milestone:
            print(f"  Milestone: {spec.milestone}")
        print(f"  Body:\n{spec.body[:400]}{'...' if len(spec.body)>400 else ''}")
        return

    url = get_repo_api_base(repo) + "/issues"
    r = requests.post(url, headers=gh_headers(token), json=payload)
    r.raise_for_status()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Create GitHub issues from project markdown docs.")
    p.add_argument("--repo", required=True, help="GitHub repo in 'owner/name' format")
    p.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable name holding the GitHub token")
    p.add_argument("--dry-run", action="store_true", help="Do not call GitHub API; just print planned issues")
    p.add_argument("--assignee", action="append", help="Assign created issues to this username (can repeat)")
    p.add_argument("--labels", action="append", help="Extra labels to add to every created issue (can repeat)")
    p.add_argument("--milestones-per-phase", action="store_true", help="Create a milestone for each Phase and attach corresponding issues")
    p.add_argument("--force", action="store_true", help="Create even if an issue with the same title exists")
    p.add_argument("--files", nargs="*", default=["development_plan.md", "data-mining-recommendations.md"], help="Markdown files to parse")

    args = p.parse_args(argv)

    token = os.getenv(args.token_env)
    if not args.dry_run and not token:
        print(f"Error: GitHub token not found in env {args.token_env}. Use --dry-run to preview.", file=sys.stderr)
        return 2

    all_specs: List[IssueSpec] = []

    # Parse each file
    for path in args.files:
        if not os.path.exists(path):
            print(f"Warning: file not found: {path}")
            continue
        text = load_text(path)
        file_ref = path
        if os.path.basename(path).lower().startswith("development_plan"):
            all_specs.extend(parse_development_plan(text, file_ref))
        elif os.path.basename(path).lower().startswith("data-mining-recommendations"):
            all_specs.extend(parse_recommendations(text, file_ref))
        else:
            # Optional: look for explicit TODO bullets in other files
            for m in re.finditer(r"^\s*-\s*\[ \]\s*(.+)$", text, re.M):
                task = m.group(1).strip()
                all_specs.append(IssueSpec(title=f"[TODO] {task}", body=f"From {file_ref}", labels=["Todo"]))

    # Apply global labels/assignees
    if args.labels:
        for spec in all_specs:
            spec.labels.extend(args.labels)
    if args.assignee:
        for spec in all_specs:
            spec.assignees = list(args.assignee)

    # Deduplicate by title within run
    unique: Dict[str, IssueSpec] = {}
    for spec in all_specs:
        if spec.title not in unique:
            unique[spec.title] = spec
    specs = list(unique.values())

    # Prepare labels
    phase_labels = sorted({l for spec in specs for l in spec.labels if l.lower().startswith("phase ")})
    priority_labels = sorted({l for spec in specs if any(lb.lower().startswith("priority ") for lb in spec.labels) for l in spec.labels})
    base_labels = set(l for spec in specs for l in spec.labels)

    if not args.dry_run and requests is None:
        print("Error: requests package not available. Install requests or run with --dry-run.", file=sys.stderr)
        return 3

    # Ensure labels exist
    if token and requests is not None:
        ensure_labels(args.repo, token, sorted(base_labels), args.dry_run)

    # Prepare milestones per phase if requested
    milestone_numbers: Dict[str, int] = {}
    if args.milestones_per_phase and token and requests is not None:
        # Determine phases
        phases = sorted({l for l in base_labels if l.lower().startswith("phase ")})
        for ph in phases:
            num = get_or_create_milestone(args.repo, token, ph, args.dry_run)
            if num is not None:
                milestone_numbers[ph] = num
        # Attach milestone names to matching specs
        for spec in specs:
            for l in spec.labels:
                if l in milestone_numbers:
                    spec.milestone = l
                    break

    # Idempotency: fetch existing issues and skip same titles
    existing_titles: Dict[str, int] = {}
    if token and requests is not None and not args.force:
        existing_titles = list_existing_issues(args.repo, token)

    created = 0
    skipped = 0
    for spec in specs:
        if not args.force and spec.title in existing_titles:
            skipped += 1
            if args.dry_run:
                print(f"[DRY-RUN] Skipping existing: {spec.title}")
            continue
        create_issue(args.repo, token or "", spec, milestone_numbers, args.dry_run)
        created += 1

    print(f"Done. Created: {created}, Skipped: {skipped}, Total planned: {len(specs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
