"""
Cron-based LLM job scheduler for pythonclaw.

Two sources of jobs
-------------------
1. Static jobs   — defined in context/cron/jobs.yaml (human-configured)
2. Dynamic jobs  — added at runtime by the Agent via cron_add / cron_remove
                   tool calls; persisted to context/cron/dynamic_jobs.json

Session isolation
-----------------
Each job gets its own persistent session via the shared SessionManager:
    session_id = "cron:{job_id}"

This means:
  - Jobs don't share context with each other or with user conversations.
  - The same job accumulates history across multiple runs.
  - Sessions can be reset via SessionManager.reset("cron:{job_id}").

Agent cron tools
----------------
Expose these to the Agent via agent.py:
    cron_add(job_id, cron, prompt, deliver_to_chat_id=None)
    cron_remove(job_id)
    cron_list()

jobs.yaml format
----------------
jobs:
  - id: daily_summary
    cron: "0 9 * * *"
    prompt: "Summarise my tasks and memory for today."
    deliver_to: telegram
    chat_id: 123456789
    enabled: true
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from ..channels.telegram_bot import TelegramBot
    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)

def _cron_dir() -> str:
    from .. import config as _cfg
    return os.path.join(str(_cfg.PYTHONCLAW_HOME), "context", "cron")


def _dynamic_jobs_file() -> str:
    return os.path.join(_cron_dir(), "dynamic_jobs.json")


def _default_jobs_path() -> str:
    return os.path.join(_cron_dir(), "jobs.yaml")


class CronScheduler:
    """
    Loads job definitions from YAML and schedules them with APScheduler.

    Each job runs inside its own session ("cron:{job_id}") managed by the
    shared SessionManager, keeping job context isolated and persistent.
    """

    def __init__(
        self,
        session_manager: "SessionManager",
        jobs_path: str | None = None,
        telegram_bot: "TelegramBot | None" = None,
    ) -> None:
        self._sm = session_manager
        self._jobs_path = jobs_path or _default_jobs_path()
        self._telegram_bot = telegram_bot
        self._scheduler = AsyncIOScheduler()

    # ── YAML loading ─────────────────────────────────────────────────────────

    def _load_jobs(self) -> list[dict]:
        if not os.path.exists(self._jobs_path):
            logger.info("[CronScheduler] No jobs file found at %s — skipping.", self._jobs_path)
            return []
        with open(self._jobs_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("jobs", [])

    # ── Job execution ────────────────────────────────────────────────────────

    async def _run_job(
        self,
        job_id: str,
        prompt: str,
        deliver_to: str | None,
        chat_id: int | None,
    ) -> None:
        session_id = f"cron:{job_id}"
        logger.info("[CronScheduler] Running job '%s' (session='%s')", job_id, session_id)

        agent = self._sm.get_or_create(session_id)
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, agent.chat, prompt)
            logger.info("[CronScheduler] Job '%s' completed.", job_id)
        except Exception as exc:
            logger.exception("[CronScheduler] Job '%s' failed: %s", job_id, exc)
            response = f"[Cron job '{job_id}' failed]\n{exc}"

        if deliver_to == "telegram" and chat_id and self._telegram_bot:
            try:
                header = f"📋 Cron job: {job_id}\n\n"
                await self._telegram_bot.send_message(chat_id, header + (response or ""))
            except Exception as exc:
                logger.error(
                    "[CronScheduler] Failed to deliver job '%s' to Telegram: %s", job_id, exc
                )

    # ── Scheduler lifecycle ──────────────────────────────────────────────────

    def load_and_register_jobs(self) -> int:
        """Parse jobs.yaml and register enabled jobs with APScheduler. Returns count."""
        jobs = self._load_jobs()
        registered = 0
        for job in jobs:
            job_id = job.get("id", "unnamed")
            if not job.get("enabled", True):
                logger.info("[CronScheduler] Skipping disabled job '%s'", job_id)
                continue

            cron_expr = job.get("cron")
            prompt = job.get("prompt")
            if not cron_expr or not prompt:
                logger.warning(
                    "[CronScheduler] Job '%s' is missing 'cron' or 'prompt' — skipped.", job_id
                )
                continue

            deliver_to = job.get("deliver_to")
            chat_id = job.get("chat_id")

            trigger = _parse_cron(cron_expr)
            self._scheduler.add_job(
                self._run_job,
                trigger=trigger,
                id=job_id,
                kwargs={
                    "job_id": job_id,
                    "prompt": prompt,
                    "deliver_to": deliver_to,
                    "chat_id": chat_id,
                },
                replace_existing=True,
            )
            logger.info(
                "[CronScheduler] Registered job '%s' (session='cron:%s') cron='%s'",
                job_id, job_id, cron_expr,
            )
            registered += 1

        return registered

    def start(self) -> None:
        """Start the APScheduler background scheduler (static + dynamic jobs)."""
        static_count = self.load_and_register_jobs()
        dynamic_count = self._register_dynamic_jobs()
        total = static_count + dynamic_count
        if total == 0:
            logger.info("[CronScheduler] No jobs to schedule — scheduler will start but be idle.")
        self._scheduler.start()
        logger.info(
            "[CronScheduler] Scheduler started: %d static + %d dynamic job(s).",
            static_count, dynamic_count,
        )

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[CronScheduler] Scheduler stopped.")

    def reload_jobs(self) -> int:
        """Hot-reload static jobs from the YAML file without stopping the scheduler."""
        self._scheduler.remove_all_jobs()
        return self.load_and_register_jobs()

    # ── Dynamic job management (called by Agent cron tools) ──────────────────

    def _load_dynamic_jobs(self) -> dict[str, dict]:
        """Load persisted dynamic jobs from JSON. Returns {job_id: job_dict}."""
        djf = _dynamic_jobs_file()
        if not os.path.exists(djf):
            return {}
        try:
            with open(djf, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[CronScheduler] Failed to load dynamic jobs: %s", exc)
            return {}

    def _save_dynamic_jobs(self, jobs: dict[str, dict]) -> None:
        djf = _dynamic_jobs_file()
        os.makedirs(os.path.dirname(djf), exist_ok=True)
        with open(djf, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)

    def _register_dynamic_jobs(self) -> int:
        """Register all persisted dynamic jobs with the scheduler."""
        jobs = self._load_dynamic_jobs()
        registered = 0
        for job_id, job in jobs.items():
            try:
                self._scheduler.add_job(
                    self._run_job,
                    trigger=_parse_cron(job["cron"]),
                    id=job_id,
                    kwargs={
                        "job_id": job_id,
                        "prompt": job["prompt"],
                        "deliver_to": job.get("deliver_to"),
                        "chat_id": job.get("chat_id"),
                    },
                    replace_existing=True,
                )
                registered += 1
                logger.info("[CronScheduler] Restored dynamic job '%s'", job_id)
            except Exception as exc:
                logger.error("[CronScheduler] Failed to restore dynamic job '%s': %s", job_id, exc)
        return registered

    def add_dynamic_job(
        self,
        job_id: str,
        cron_expr: str,
        prompt: str,
        deliver_to: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """
        Add a new dynamic job (called from the Agent cron_add tool).
        Persists to dynamic_jobs.json so it survives restarts.
        """
        try:
            trigger = _parse_cron(cron_expr)
        except ValueError as exc:
            return f"Invalid cron expression: {exc}"

        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            id=job_id,
            kwargs={
                "job_id": job_id,
                "prompt": prompt,
                "deliver_to": deliver_to,
                "chat_id": chat_id,
            },
            replace_existing=True,
        )

        jobs = self._load_dynamic_jobs()
        jobs[job_id] = {
            "cron": cron_expr,
            "prompt": prompt,
            "deliver_to": deliver_to,
            "chat_id": chat_id,
        }
        self._save_dynamic_jobs(jobs)
        logger.info("[CronScheduler] Added dynamic job '%s' (cron='%s')", job_id, cron_expr)
        return f"Job '{job_id}' scheduled: runs '{cron_expr}'. Session: cron:{job_id}."

    def remove_dynamic_job(self, job_id: str) -> str:
        """Remove a dynamic job (called from the Agent cron_remove tool)."""
        jobs = self._load_dynamic_jobs()
        if job_id not in jobs and not self._scheduler.get_job(job_id):
            return f"Job '{job_id}' not found."
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        jobs.pop(job_id, None)
        self._save_dynamic_jobs(jobs)
        logger.info("[CronScheduler] Removed dynamic job '%s'", job_id)
        return f"Job '{job_id}' removed."

    def list_jobs(self) -> str:
        """Return a human-readable list of all active jobs (called from cron_list tool)."""
        scheduler_jobs = self._scheduler.get_jobs()
        dynamic = self._load_dynamic_jobs()
        if not scheduler_jobs:
            return "No scheduled jobs."
        lines = []
        for job in scheduler_jobs:
            tag = "[dynamic]" if job.id in dynamic else "[static]"
            next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M %Z") if job.next_run_time else "paused"
            lines.append(f"  {tag} {job.id} | next: {next_run}")
        return "Active cron jobs:\n" + "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_cron(expr: str) -> CronTrigger:
    """Convert a 5-field cron expression string into an APScheduler CronTrigger."""
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression (expected 5 fields): '{expr}'")
    minute, hour, day, month, day_of_week = parts
    return CronTrigger(
        minute=minute,
        hour=hour,
        day=day,
        month=month,
        day_of_week=day_of_week,
    )
