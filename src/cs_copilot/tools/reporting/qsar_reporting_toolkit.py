#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for canonical QSAR report payload and LaTeX export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.agents.qsar_report_payload import (
    add_bullets_block,
    add_files_block,
    add_kv_block,
    add_paragraph_block,
    add_section,
    add_table_block,
    init_report_payload,
)

from .qsar_latex import write_latex_report, write_payload_json


def _get_report_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("qsar_report", {})
    state.setdefault("last_request", {})
    state.setdefault("last_result", {})
    return state


class QSARReportingToolkit(Toolkit):
    def __init__(self):
        super().__init__("qsar_reporting")
        self.register(self.init_qsar_report_payload)
        self.register(self.append_qsar_report_section)
        self.register(self.export_qsar_latex_report)

    def init_qsar_report_payload(
        self,
        *,
        report_type: str,
        title: str,
        intro: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        payload = init_report_payload(
            report_type=report_type,
            title=title,
            intro=intro,
            metadata=metadata or {},
        )
        if agent is not None:
            state = _get_report_state(agent)
            state["last_request"] = {
                "report_type": report_type,
                "title": title,
            }
            state["last_result"]["report_payload"] = payload
        return payload

    def append_qsar_report_section(
        self,
        *,
        payload: Dict[str, Any],
        section_title: str,
        blocks: List[Dict[str, Any]],
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        section = add_section(payload, title=section_title)
        for block in blocks:
            block_type = block.get("type")
            if block_type == "paragraph":
                add_paragraph_block(section, title=block.get("title", ""), text=block.get("text", ""))
            elif block_type == "bullets":
                add_bullets_block(section, title=block.get("title", ""), items=block.get("items", []))
            elif block_type == "table":
                add_table_block(
                    section,
                    title=block.get("title", ""),
                    columns=block.get("columns", []),
                    rows=block.get("rows", []),
                )
            elif block_type == "kv_list":
                add_kv_block(section, title=block.get("title", ""), items=block.get("items", []))
            elif block_type == "files":
                add_files_block(section, title=block.get("title", ""), items=block.get("items", []))
        if agent is not None:
            _get_report_state(agent)["last_result"]["report_payload"] = payload
        return payload

    def export_qsar_latex_report(
        self,
        *,
        payload: Dict[str, Any],
        output_dir: Optional[str] = None,
        basename: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        root = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else (Path(".files") / "qsar_reports").resolve()
        )
        root.mkdir(parents=True, exist_ok=True)
        name = basename or f"{payload.get('report_type', 'qsar_report')}_report"
        tex_path = root / f"{name}.tex"
        json_path = root / f"{name}.payload.json"
        latex_result = write_latex_report(payload, str(tex_path))
        payload_result = write_payload_json(payload, str(json_path))
        result = {
            **latex_result,
            **payload_result,
        }
        if agent is not None:
            state = _get_report_state(agent)
            state["last_result"]["report_payload"] = payload
            state["last_result"]["latex_report_path"] = result["report_path"]
            state["last_result"]["report_payload_path"] = result["payload_path"]
        return result

