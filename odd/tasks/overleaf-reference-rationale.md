# Overleaf reference-rationale correction

## Objective and authorization
Apply only the three approved manuscript changes for reviewer comment 3 to Overleaf project 6a96fbafea5f6a7dbd332ab1 through its Git integration.
User explicitly authorized clone, edit and push using locally configured Overleaf Git authentication after being warned about tracked-change/comment displacement.

## Scope and constraints
- Clone target: https://git.overleaf.com/6a96fbafea5f6a7dbd332ab1.
- Isolated checkout: resultados/overleaf/6a96fbafea5f6a7dbd332ab1, ignored by the parent repository.
- Add operational rationale and limitations after the ramp/plateau definition.
- Correct slope description to distinguish fixed and automatic modes with the implemented range floor.
- Qualify distance/sign interpretation rather than asserting guaranteed classification.
- Match the actual remote manuscript before editing; do not overwrite it with the local manuscript.
- Preserve figures, results, references and unrelated collaborator edits. No invented ablation results, no reviewer checkbox completion.
- Stop if configured noninteractive authentication is unavailable; never request secrets in chat or discover unrelated credentials.
- No force push. Recheck remote before publication and preserve any concurrent changes.
- Existing local paper files and rejected GitHub push are separate, untouched work.

## Tasks and checks
- [ ] O01 — Clone and identify the remote manuscript and baseline commit.
  - Verify target, clean checkout, actual reference section and matching edit anchors.
- [ ] O02 — Apply the three localized English LaTeX corrections and verify them.
  - Check exact diff scope, preserved equations/labels and balanced environments; compile if an available local TeX toolchain supports the project.
- [ ] O03 — Commit and push the verified correction to Overleaf.
  - Conventional commit without attribution; inspect remote divergence before push; verify remote commit and local cleanliness afterward.

## Workflow
Direct single-manuscript edit. TDD is not applicable to prose; use diff and LaTeX checks. No native RDD enabled; ordinary verification only. CodeGraph is not needed for manuscript content editing and previously failed in the parent repository.

## Progress and next step
Authorization recorded; remote content/authentication not yet verified. Attempt noninteractive configured Git clone, stopping on authentication failure.

## Recovery locator
Repository-relative path: odd/tasks/overleaf-reference-rationale.md. Engram topic: odd/overleaf-reference-rationale/tasks. Do not upload this local recovery document into the Overleaf manuscript project.
