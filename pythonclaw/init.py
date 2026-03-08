"""
Project initialiser for pythonclaw.

Creates a `context/` directory with the required sub-directories and copies
template files so the agent has a ready-to-use starting point.

Sub-directories created
-----------------------
  context/memory/     — MEMORY.md + daily YYYY-MM-DD.md logs
  context/knowledge/  — .txt / .md knowledge-base files for RAG
  context/skills/     — skill directories (copied from templates)
  context/persona/    — persona .md files
  context/soul/       — SOUL.md identity document
  context/tools/      — TOOLS.md local environment notes
  context/files/      — downloaded / generated files
"""

from __future__ import annotations

import os
import shutil


def init(project_path: str | None = None) -> None:
    """
    Initialise a new PythonClaw project.

    Copies template files into ``<project_path>/context/`` only for
    directories that do not already exist (safe to re-run).
    Defaults to ``~/.pythonclaw``.
    """
    if project_path is None:
        from . import config
        application_dir = str(config.PYTHONCLAW_HOME)
    else:
        application_dir = os.path.abspath(project_path)
    context_dir = os.path.join(application_dir, "context")
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(pkg_dir, "templates")

    print(f"Initialising PythonClaw in: {application_dir}")

    if not os.path.exists(templates_dir):
        print("Warning: templates directory not found — using minimal fallbacks.")

    for component in ("memory", "knowledge", "skills", "persona", "soul", "tools", "files"):
        target = os.path.join(context_dir, component)
        source = os.path.join(templates_dir, component)

        if os.path.exists(target):
            print(f"  - {target} already exists, skipping.")
            continue

        if os.path.isdir(source):
            shutil.copytree(source, target)
            print(f"  - Created {target} (from template)")
        else:
            os.makedirs(target, exist_ok=True)
            print(f"  - Created {target} (empty)")
            # Minimal fallback content for directories with no template
            if component == "memory":
                with open(os.path.join(target, "MEMORY.md"), "w") as f:
                    f.write("# Long-Term Memory\n")
            elif component == "knowledge":
                with open(os.path.join(target, "README.txt"), "w") as f:
                    f.write("Add your knowledge-base .txt files here.\n")

    print("\nInitialisation complete. Start your agent with this context directory.")
