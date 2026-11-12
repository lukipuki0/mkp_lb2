---
name: professional-lucid-diagrams
description: "Trigger: Lucid, professional diagrams, diagramas, diagram batches, print legibility. Create source-faithful diagrams with prototype approval and visual verification."
license: Apache-2.0
metadata:
  author: danilo
  version: "1.0"
---

## Activation Contract

Use for creating, revising, reviewing, or delivering professional Lucid diagrams. Load the workflow and visual references before acting; load the integration reference only for a requested LaTeX/Overleaf destination.

## Hard Rules

- Prefer Lucid. Preserve source semantics, provenance, and current user edits; never silently change topology, multiply pages, or change delivery mode.
- Inspect available capabilities and fresh provider schemas/resources. An existing SVG requires the provider conversion route; missing conversion is a blocker, not permission to reconstruct it.
- Approve one representative prototype before a multi-diagram batch. Apply project-specific branding, not an assumed universal palette.
- Require structural coverage AND rendered target-size visual evidence. Never equate creation, export, integration, or push with print readiness.
- Ask one question at a time, then wait. Never publish, share permissions, commit, push, install, or change configuration without explicit authorization.

## Decision Gates

| Condition | Action |
| --- | --- |
| Required source missing or conflicting | Request the missing source or resolve precedence; do not invent content. |
| Concrete ordinary creation | Proceed with sensible stated defaults. |
| Collaborative planning or batch | Confirm plan; create one prototype, obtain approval, then produce the coherent batch. |
| Density requires general/detail split or more pages | Propose the smallest faithful split and obtain approval first. |
| TikZ or LaTeX destination absent | Do not experiment with TikZ or perform integration. |

## Execution Steps

1. Inspect sources, revisions, capabilities, destination dimensions, and project brand evidence.
2. Inventory semantic nodes/edges and agree on page roles; separate repeated reference anchors from unique nodes.
3. Build and review the prototype when required; carry approved visual conventions across the batch.
4. Verify coverage, render every final page at target size, and recheck siblings after shared edits. Bound corrections and report unresolved defects.
5. Integrate only if requested. Return a versioned manifest and evidence-specific status; leave unauthorized publication pending.

## Output Contract

Return a short delivery with actual links, document/page roles, file paths, verification evidence, and pending steps. Distinguish created, structurally verified, visually reviewed at target size, integrated, and compiled-and-inspected.

## References

- [Lucid workflow](references/lucid-workflow.md)
- [Visual quality](references/visual-quality.md)
- [Optional LaTeX/Overleaf](references/latex-overleaf.md)
