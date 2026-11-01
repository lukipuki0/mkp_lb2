---
name: presentacion-beamer
description: "Trigger: presentación, presentacion, beamer, conferencia, seminario, defensa, slides científicas, simple, clara, no cargada. Sparse Spanish Beamer for a committee; ask duration before writing; title slide required; one short formula only when it is the frame."
license: Apache-2.0
metadata:
  author: gentleman-programming
  version: "1.2"
---

## Activation Contract

Use for a scientific talk in LaTeX Beamer: conferencia, seminario de título, or defensa.

Do not use for PowerPoint, Google Slides, HTML, posters, or non-scientific decks. Do not load `scientific-slides` image generation or its Madrid template when this skill matches.

## Hard Rules

- Copy `assets/preamble.tex` unchanged. Do not restyle colors, theme, fonts, or citation setup.
- Default audience is the committee and advisors. Define each technical term the first time it appears. No paper jargon.
- Spanish slide prose. A frame title is one short claim, not a label such as "Introducción".
- One idea per frame, readable in one glance. Support it with exactly one of: three short lines, one block, one figure, or a table of at most three rows. Do not combine them.
- A short formula is allowed only when that formula is the idea of the frame. That frame has no other number and no derivation.
- If the idea does not fit in one sentence, split the frame. Never shrink fonts, margins, or spacing to fit more text.
- Leave the lower half empty when the claim is enough. The spoken explanation stays off the slide.
- No TikZ timelines, multi-panel diagrams, equation derivations, or two-column layouts unless the user asks for that frame.
- `presentacion_conferencia/main.tex` is the color and theme source only. Do not copy its density.
- Cite only with `\smallcite{keys}` at the frame bottom. Never invent keys, numbers, or results; mark a gap `\todo{...}`.
- The title frame is required and `[plain]`. Include `fig/logo_100.png` and `fig/Logo_pucv_ok.PNG` only when both exist. Always name the advisors on that frame.
- Do not add a TOC, section dividers, a contributions frame, a references frame, or one figure per section unless the user asked for that item.
- Section dividers, when requested, use the plain divider in `assets/deck-skeleton.tex`.
- Set `\talkfooter` to this talk. Do not leave `Tema de la charla`.
- Default identity, override only if the user names another: Lucas Erazo; Escuela de Ingeniería Informática, PUCV; guía Emanuel Vega; correferente Ricardo Soto.
- Write `presentacion_<slug>/main.tex`, `referencias.bib`, and `fig/`. Never overwrite `presentacion_conferencia/` or `presentacion_seminario/`.

## Decision Gates

| Situation | Action |
| --- | --- |
| Duration not answered this turn | Stop. Propose conferencia corta: problema, brecha, método, un resultado, cierre. Do not write the `.tex`. |
| User confirms conferencia corta | Those five beats only. No TOC, no plan, no extra closing frames. |
| User asks for seminario or avance | TOC and numbered sections. Work plan only if the user asked for it. |
| User asks for defensa | Same visual contract. Add hypothesis, validation, limitations, and a questions frame. |
| Frame would hold two ideas or two visuals | Split it. Do not tighten the layout. |
| Missing figure or datum | Leave `\todo`. Do not draw a fake result. |
| `scientific-slides` also matches | This skill wins for Beamer in this repo. |

## Execution Steps

1. If this turn did not already fix the duration, ask once and stop. Propose conferencia corta. Do not write files yet.
2. Read `assets/preamble.tex` and `assets/deck-skeleton.tex`.
3. Read only the source the user names. Extract claims that source states.
4. Fill the skeleton. Keep `\smallcite` keys that exist in `referencias.bib`. Split any frame that fails the one-glance rule before compiling.
5. Run `latexmk -pdf -interaction=nonstopmode main.tex` when `latexmk` exists. If it does not, report the missing tool and do not claim a PDF.
6. Return the deck path, frame count, compile result, and every `\todo`.

## Output Contract

Return:
- Deck path and frame count.
- Compile command and result, or the missing tool.
- Which claims came from which source, and any `\todo` left.
- Confirmation that `presentacion_conferencia/` was not modified.

## References

- `assets/preamble.tex` — visual contract from `presentacion_conferencia/main.tex`.
- `assets/deck-skeleton.tex` — required frame order.
- `presentacion_conferencia/main.tex` — source deck. Read it. Do not overwrite it.
