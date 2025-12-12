# Thesis Project Guidelines

> **CRITICAL**: When appending to `CHANGELOG.md`, always include the current timestamp (date and time).



## P1 - Critical (Every Session)

### 1. Requirement Compliance
- **Verify against Guidelines**: Ensure content aligns with `UP Thesis Guidelines` and `UP Outline`.
- **Check Constraints**: Monitor word counts, formatting requirements, and required sections (e.g., Abstract, Introduction types).

### 2. Fact Verification

#### 2.1 External Verification
- **Always externally verify factual claims** before marking as correct
- Use web search to confirm dates, names, percentages, and events
- Compare document claims against authoritative sources

#### 2.2 Source Priority
Prioritize sources in this order:
1. Government sources (texas.gov, austintexas.gov, capitol.texas.gov)
2. Academic sources (university research)
3. Established news organizations (Texas Tribune, KUT)
4. Official organizational sites

#### 2.3 Verification Steps
1. Search for claims in target document
2. Check if cited reference exists in references.bib
3. Externally verify factual accuracy via web search
4. Compare authoritative sources vs current citations
5. Update bib entries or inline citations as needed

### 3. Citation Integrity

#### 3.1 Citation Checking
- Verify inline `\cite{key}` commands have matching bib entries
- Search using multiple keywords (author name, title words, year)

#### 3.2 Adding New Entries
- Use verified external sources
- Note the verification date in comments

---

## P2 - Important (Regular Maintenance)

### 3. Bibliography Management

#### 3.1 Formatting Standard
```bibtex
@article{key,

  author = {Name},

  year = {2025},

  title = {Title}

}
```
- Blank line after `@type{key,`
- Blank line between each field
- Two-space indentation for fields

#### 3.2 Organization
Use commented section headers:
- Primary Sources (Government, Court)
- Secondary Sources (Academic)
- News Sources
- Organizational Sources
- Data Sources

### 4. Documentation

#### 4.1 Session Tracking
- **Prompt Logging**: Append conversation prompts (or a summary) to `PROMPTS-LOG.md` in the root directory.
- **Timestamping**: Every new insertion in `PROMPTS-LOG.md` MUST be preceded by a header with the current timestamp (e.g., `## YYYY-MM-DDTHH:MM:SS - Session Description`).
- **Full Fidelity**: Never truncate or use ellipses (...). Log the exact full text of the prompt to ensure complete context is preserved.
- **Consistent Structure**: Use a numbered list for each prompt turn (e.g., `1. "Prompt text"`).
- **Major Decisions**: Record major decisions in `CHANGELOG.md` with date and time.
- **Append ONLY**: Do not revise existing logs; add new entries at the end.
- **Review Status**: Mark historical entries as `(REVIEWED)` and new entries as `(UNREVIEWED)`
- Create a `TODO` item for reviewing UNREVIEWED logs

#### 4.2 Version Control
- Commit frequently with descriptive messages
- Push to remote after each logical unit of work

### 5. TODO Management

#### 5.1 Insertion Policy
- **Context-Aware Insertion**: Do not blindly append new items (TODOs, guideline sections, or list entries) to the top or bottom of files.
- **Prioritize Immediately**: Assess the priority of the new item (P1, P2, P3, or High Priority) and insert it into the corresponding section or tier.
- **Best Prioritized Place**: Insert the item in the most logical position within its priority group (e.g., grouping similar tasks, respecting dependencies, or ordering by importance).
- **Renumbering**: If inserting into a numbered list or structured sequence (like guideline sections), update numbering to maintain consistency.
- **Maintain Structure**: Ensure the item is placed logically within its priority group relative to others.

---

## P3 - Housekeeping

#### 5.2 Completion Policy
- **Move to Archive**: When a task is done, move it from `TODO.md` to `TODO-COMPLETED.md`.
- **Maintain Structure**: Place the completed item under its corresponding Priority Header (P1, P2, P3) in `TODO-COMPLETED.md`.
- **Timestamping**: Append the completion timestamp to the item (e.g., `(Completed: YYYY-MM-DDTHH:MM)`).

### 6. File Organization

#### 6.1 Naming Conventions
| Suffix | Meaning |
|--------|---------|
| `-SUBMITTED` | Finalized/submitted work |
| `-TO_INTEGRATE` | Content pending integration |
| `-COMPREHENSIVE` | Complete/long-form documents |
| `-OLD` | Superseded/deprecated content |
| `.d/` | LaTeX projects or grouped files |
| `#-` prefix | Chronological order (e.g., `1-`, `2.1-`, `2.2-`) |
| `*.1, *.2` | Parallel documents (same base number, `.1` = likely first) |

#### 6.2 Directory Structure
```
thesis/
├── Thesis_Draft/
│   └── Thesis_Draft_Reference_Materials/
│       ├── Prompts.d/      # Session prompts by quality tier
│       ├── Background-COMPREHENSIVE.d/
│       └── references.bib
├── Assignments_and_Proposal-SUBMITTED/
├── Deprecated_Writings/    # Numbered chronologically with README
├── TODO.md
├── GUIDELINES.md
└── CHANGELOG.md
```

#### 6.3 README Requirements
- Add `README.md` to any directory with non-obvious organization
- Explain numbering schemes, content relationships, or special conventions

### 7. Safe Deletion
- Delete files only after verifying content is captured elsewhere
- Track deferred work in `TODO.md` with clear, actionable items



### 8. Session Wrap-Up
- **Log Prompts**: Ensure all recent prompts are appended to `PROMPTS-LOG.md` with a timestamp, following full-fidelity rules.
- **Update Changelog**: Add a timestamped entry to `CHANGELOG.md` summarizing key changes, decisions, and completed tasks.
- **Review TODOs**: Verify all completed work is checked off in `TODO.md` and moved to `TODO-COMPLETED.md`.
- **Final Commit**: Stage and commit all project changes with a descriptive message (e.g., `git commit -am "Session update: [Topic]"`).

