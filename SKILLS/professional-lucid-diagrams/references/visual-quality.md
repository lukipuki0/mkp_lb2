# Verify the delivered size, not the zoomed canvas

## Establish the target

- Read project brand guidance or inspect approved assets. Separate verified brand rules from proposed styling; use a neutral stated default if no brand evidence exists. Never treat an illustrative palette as an official guide or default for every project.
- Establish actual display or printed width and height. For document integration, read the real main file, full preamble, styles, margins, columns, orientation, and figure placement; fragments do not establish usable space.
- Do not assume A4 or default to A3. Recommend a configurable minimum font size appropriate to the audience and medium; a nominal 9 pt threshold is a recommendation only if proposed, never an unasked user requirement.

## Compose for recognition

Use compact balanced whitespace, intentional flow, aligned groups, and clear connector corridors. Avoid giant empty central boxes and uniformly tiny text as a density workaround. Size labels and nodes to content; keep labels unobscured. Preserve semantic shapes and distinguish dependency arrows from operational flow and state routes. Do not encode important distinctions by color alone.

If faithful content cannot fit legibly, propose an approved general/detail split instead of silently shrinking everything or expanding page count. Keep general views useful independently while labeling their coverage honestly.

## Measure and inspect

1. Record source units and export dimensions. Calculate effective size only with known units: for a PDF with nominal font size in points, `effective_pt = nominal_pt * placed_width_pt / selected_box_width_pt` for uniform width scaling. When both width/height constrain placement, use the actual limiting scale. Include cropping and page selection in the calculation.
2. Do not equate measured glyph height with nominal font points. Canvas units, raster pixels, PDF points, and LaTeX points differ; document conversions and raster resolution instead of assuming equivalence. For PNG at known DPI, `printed_inches = pixels / DPI`; pixel count alone does not prove legibility.
3. Render every final page and inspect at the real target size: text clipping, text-on-text, connector-on-text, label collisions, node overlap, crossings, arrowheads, state branches, page boundaries, contrast, and reading order. Structural inspection is separate and cannot replace this review.
4. Re-render affected pages AND sibling pages after shared style/layout edits. Record output revision, target dimensions, inspected pages, findings, and remaining limits. If rendering or target measurement is unavailable, mark visual verification pending.

## Revalidate every updated export

Inspect page count, MediaBox, CropBox, ArtBox (or documented absence), and actual content bounds for each updated PDF. Recompute each page's placement/crop from that revision. Never reuse magic trim values from an older binary or assume siblings share bounds.

Crop only evidenced empty margins. Protect text, labels, strokes, arrowheads, and intended whitespace; inspect the cropped result at target size. Retain source originals and verify replacement hashes/provenance. A PNG crop changes only that image: it does not change the Lucid source page or PDF page boxes.

Prefer vector PDF integration with explicit page selection when supported. A successful export, a large screenshot, or a font-size calculation alone does not prove final readability.
