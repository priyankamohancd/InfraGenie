"""
Top-level Phase 1 pipeline: diagram file in, 5 Terraform files written to
an output directory.

This is the single function Phase 2's API layer will call per job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from arch2terraform.adapters.registry import parse_diagram
from arch2terraform.classifier.classifier import classify_diagram
from arch2terraform.generator.hcl_generator import (
    generate_main_tf,
    generate_outputs_tf,
    generate_provider_tf,
    generate_readme_md,
    generate_variables_tf,
)
from arch2terraform.resolver.resolver import resolve_relationships
from arch2terraform.schemas.resources import ResourceGraph


@dataclass
class PipelineResult:
    output_dir: str
    files_written: list[str]
    graph: ResourceGraph
    warnings: list[str]


def run_pipeline(input_file: str, output_dir: str) -> PipelineResult:
    diagram = parse_diagram(input_file)
    classified, unclassified = classify_diagram(diagram)
    graph = resolve_relationships(diagram, classified)
    graph.unclassified_nodes = unclassified  # resolver also derives this; classifier's list is authoritative

    os.makedirs(output_dir, exist_ok=True)

    needs_random_provider = any(r.companion_blocks for r in graph.resources)

    files = {
        "provider.tf": generate_provider_tf(needs_random_provider=needs_random_provider),
        "variables.tf": generate_variables_tf(),
        "main.tf": generate_main_tf(graph),
        "outputs.tf": generate_outputs_tf(graph),
        "README.md": generate_readme_md(graph, input_file, diagram.warnings),
    }

    written: list[str] = []
    for filename, content in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(path)

    return PipelineResult(
        output_dir=output_dir,
        files_written=written,
        graph=graph,
        warnings=diagram.warnings,
    )
