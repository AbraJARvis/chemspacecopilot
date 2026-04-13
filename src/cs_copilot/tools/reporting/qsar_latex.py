#!/usr/bin/env python
# coding: utf-8
"""
Canonical QSAR LaTeX report rendering utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\geometry{margin=2.5cm}
\begin{document}
"""


def escape_latex(text: Any) -> str:
    if text is None:
        return ""
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _render_table(columns: List[str], rows: List[List[str]]) -> str:
    colspec = " | ".join(["p{3cm}"] * max(len(columns), 1))
    header = " & ".join(escape_latex(column) for column in columns) + r" \\"
    body_rows = "\n".join(
        " & ".join(escape_latex(cell) for cell in row) + r" \\"
        for row in rows
    )
    return "\n".join(
        [
            r"\begin{longtable}{" + colspec + "}",
            r"\toprule",
            header,
            r"\midrule",
            body_rows or r"\multicolumn{1}{l}{Aucune donnee disponible} \\",
            r"\bottomrule",
            r"\end{longtable}",
        ]
    )


def render_report_payload_to_latex(payload: Dict[str, Any]) -> str:
    parts: List[str] = [LATEX_PREAMBLE]
    title = escape_latex(payload.get("title", "Rapport QSAR"))
    intro = escape_latex(payload.get("intro", ""))
    metadata = payload.get("metadata", {}) or {}

    parts.extend(
        [
            rf"\title{{{title}}}",
            r"\author{ChemSpaceCopilot}",
            rf"\date{{{escape_latex(metadata.get('generated_date', metadata.get('trained_date', '')))}}}",
            r"\maketitle",
        ]
    )
    if intro:
        parts.append(intro + "\n")

    if metadata.get("final_status"):
        parts.append(
            r"\noindent\textbf{Statut final :} "
            + escape_latex(metadata["final_status"])
            + "\n"
        )

    for section in payload.get("sections", []):
        parts.append(r"\section*{" + escape_latex(section.get("title", "")) + "}")
        for block in section.get("blocks", []):
            block_title = block.get("title")
            if block_title:
                parts.append(r"\subsection*{" + escape_latex(block_title) + "}")
            block_type = block.get("type")
            if block_type == "paragraph":
                parts.append(escape_latex(block.get("text", "")) + "\n")
            elif block_type == "bullets":
                items = block.get("items", []) or []
                parts.append(r"\begin{itemize}[leftmargin=1.5em]")
                parts.extend(r"\item " + escape_latex(item) for item in items)
                parts.append(r"\end{itemize}")
            elif block_type == "kv_list":
                items = block.get("items", []) or []
                parts.append(r"\begin{itemize}[leftmargin=1.5em]")
                for key, value in items:
                    parts.append(
                        r"\item \textbf{" + escape_latex(key) + "} : " + escape_latex(value)
                    )
                parts.append(r"\end{itemize}")
            elif block_type == "table":
                parts.append(_render_table(block.get("columns", []), block.get("rows", [])))
            elif block_type == "files":
                items = block.get("items", []) or []
                parts.append(r"\begin{itemize}[leftmargin=1.5em]")
                for item in items:
                    label = escape_latex(item.get("label", "Fichier"))
                    path = escape_latex(item.get("path", ""))
                    parts.append(r"\item \textbf{" + label + "} : \texttt{" + path + "}")
                parts.append(r"\end{itemize}")

    parts.append(r"\end{document}")
    return "\n".join(parts)


def write_latex_report(payload: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report_payload_to_latex(payload))
    return {"report_path": str(destination), "report_type": payload.get("report_type")}


def write_payload_json(payload: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {"payload_path": str(destination), "report_type": payload.get("report_type")}

