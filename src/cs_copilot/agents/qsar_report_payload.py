#!/usr/bin/env python
# coding: utf-8
"""
Helpers for building canonical QSAR report payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def init_report_payload(
    *,
    report_type: str,
    title: str,
    intro: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "report_type": report_type,
        "title": title,
        "intro": intro,
        "metadata": metadata or {},
        "sections": [],
        "files": [],
    }


def add_section(payload: Dict[str, Any], *, title: str) -> Dict[str, Any]:
    section = {"title": title, "kind": "section", "blocks": []}
    payload.setdefault("sections", []).append(section)
    return section


def add_paragraph_block(section: Dict[str, Any], *, title: str, text: str) -> None:
    section.setdefault("blocks", []).append({"type": "paragraph", "title": title, "text": text})


def add_bullets_block(section: Dict[str, Any], *, title: str, items: List[str]) -> None:
    section.setdefault("blocks", []).append({"type": "bullets", "title": title, "items": items})


def add_table_block(
    section: Dict[str, Any], *, title: str, columns: List[str], rows: List[List[str]]
) -> None:
    section.setdefault("blocks", []).append(
        {"type": "table", "title": title, "columns": columns, "rows": rows}
    )


def add_kv_block(section: Dict[str, Any], *, title: str, items: List[List[str]]) -> None:
    section.setdefault("blocks", []).append({"type": "kv_list", "title": title, "items": items})


def add_files_block(section: Dict[str, Any], *, title: str, items: List[Dict[str, str]]) -> None:
    section.setdefault("blocks", []).append({"type": "files", "title": title, "items": items})

