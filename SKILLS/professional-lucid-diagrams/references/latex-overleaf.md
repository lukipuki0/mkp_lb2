# Integrate only into a requested destination

## Authorization and access

- Load this reference only when LaTeX/Overleaf delivery is requested. Diagram approval does not authorize integration, external sharing, commits, or pushes.
- Inspect available integration capabilities. Do not claim an Overleaf MCP exists without discovering it. If authorized and available, use Overleaf's official Git integration in a safe separate clone root, not inside another repository. Keep worktrees needing CodeGraph under the user's home with their own index.
- Let the human authenticate through a secure credential store. Never ask for tokens in chat or put them in commands, URLs, environment logs, or plaintext `credential.helper store`. Do not install tools or change configuration without separate authorization.
- If identity is missing, ask for it; never guess. Use explicitly supplied per-command identity when appropriate, not persistent Git configuration unless authorized.

## Integrate against current source

1. Inspect repository status and current remote/source revision. Read the latest main file, styles, and annex files with actual names/extensions, including `.tex` versus `.txt`. Preserve unrelated changes, tables, prose, and source external originals.
2. Derive usable figure dimensions from the real document configuration. Prefer vector PDFs with explicit `page=N` selection rather than unnecessary splitting. Keep filenames, page roles, labels, and source revisions aligned with the manifest.
3. Include shared macros exactly once and declare package/file dependencies explicitly. Do not copy an uncompiled giant helper or force an appendix numbering reset. Integrate before closing the intended chapter.
4. Ensure label uniqueness and reference resolution. Use `pageref` for unnumbered custom headers when a numbered `ref` would mislead. Captions must state whether coverage is partial, general, detail, or collectively complete across multiple pages.
5. Compile with the actual project entry point and available toolchain; inspect logs and every affected final page at target size. Check sibling figures, missing files, unresolved references, clipping, and overflow. Do not replace missing PDFs with placeholders and call the document ready.

## Publication gate

Obtain explicit consent for commit and push; general workflow or integration approval is insufficient. If declined, stop at authorized local changes and report publication pending without retries.

Before any authorized commit, inspect status, diff, and recent history; stage exact intended files and no secrets. Preserve unrelated work: no blind reset, clean, or stash. Before authorized push, recheck the remote and divergence; use non-force push only. Resolve conflicts through an approved plan, not destructive shortcuts.

A push does not prove Overleaf compiled successfully. Validate the remote compilation/report and inspect its rendered output when access permits; otherwise explicitly report remote compile/visual validation pending. Local compilation or static checks cannot substitute for that evidence. Never change sharing permissions merely because a diagram or document was created.
