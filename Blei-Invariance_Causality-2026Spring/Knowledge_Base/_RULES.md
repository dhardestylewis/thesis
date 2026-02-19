# Knowledge Base Maintenance Rules
*Strict protocols for updating the "Intelligence Center".*

## 1. The No-Loss Principle
**Rule:** When merging files, content must be **additive**.
*   Never delete a "Concept" without a specific instruction.
*   Never remove a "Citation" unless it is proven wrong.
*   If simplifying structure, **move** content, do not delete it.

## 2. Granularity & Grounding
**Rule:** Every entry must be **grounded** in a specific text.
*   **Good:** "Defined in [Peters et al., 2016, p. 2]"
*   **Bad:** "Defined generally as..."
*   **Preservation:** Do not summarize away specific page numbers or equation references.

## 3. Atomic Updates
**Rule:** Update the Knowledge Base **immediately** after processing a text.
*   Do not wait for "later cleanup."
*   If reading Pearl Ch 3, update `Reference_Atlas.md` with Ch 3 terms *during* the reading session.

## 4. Versioning Protocol
**Rule:** Create timestamped snapshots before major edits or report generation.
*   **Reader Reports:** Always create a `_timestamp.tex` copy before final compilation.
*   **Backups:** `Knowledge_Base/Backup_YYYYMMDD/` should be used for major structural refactors.

## 5. File Structure
*   **Glossary.md:** Definitions, Notation, Terminology. [Dictionary]
*   **Reference_Atlas.md:** Bibliographic Index, Concept Library, Project Map, Canonical Examples. [Encyclopedia]
*   **Project_Log.md:** Immutable record of major decisions. [History]

## 6. The "Toolbox" Taxonomy
**Rule:** Explicitly categorize methods in the Glossary.
*   **Graphical Tool:** (e.g., Backdoor Criterion) logic for identification.
*   **Algebraic Tool:** (e.g., Adjustment Formula) math for estimation.
