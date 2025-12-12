# Thesis Project Guidelines

> **CRITICAL**: When appending to `CHANGELOG.md`, always include the current timestamp (date and time).

> **CRITICAL**: To find pending TODOs or session context, refer to `Prompts.d/` files (Highest_Quality, Medium_Quality) and the current conversation—do NOT search across the codebase.

## P1 - Critical (Every Session)

### 1. Fact Verification

#### 1.1 External Verification
- **Always externally verify factual claims** before marking as correct
- Use web search to confirm dates, names, percentages, and events
- Compare document claims against authoritative sources

#### 1.2 Source Priority
Prioritize sources in this order:
1. Government sources (texas.gov, austintexas.gov, capitol.texas.gov)
2. Academic sources (university research)
3. Established news organizations (Texas Tribune, KUT)
4. Official organizational sites

#### 1.3 Verification Steps
1. Search for claims in target document
2. Check if cited reference exists in references.bib
3. Externally verify factual accuracy via web search
4. Compare authoritative sources vs current citations
5. Update bib entries or inline citations as needed

### 2. Citation Integrity

#### 2.1 Citation Checking
- Verify inline `\cite{key}` commands have matching bib entries
- Search using multiple keywords (author name, title words, year)

#### 2.2 Adding New Entries
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
- Append conversation prompts to `Prompts.d/Prompts-COMPREHENSIVE.txt`
- Categorize prompts by quality tier (Highest_Quality, Medium_Quality, etc.)
- Record session changes in `CHANGELOG.md` with date and time
- **Append ONLY**: Do not revise existing logs; add new entries at the end
- **Review Status**: Mark historical entries as `(REVIEWED)` and new entries as `(UNREVIEWED)`
- Create a `TODO` item for reviewing UNREVIEWED logs

#### 4.2 Version Control
- Commit frequently with descriptive messages
- Push to remote after each logical unit of work

---

## P3 - Housekeeping

### 5. File Organization

#### 5.1 Naming Conventions
| Suffix | Meaning |
|--------|---------|
| `-SUBMITTED` | Finalized/submitted work |
| `-TO_INTEGRATE` | Content pending integration |
| `-COMPREHENSIVE` | Complete/long-form documents |
| `-OLD` | Superseded/deprecated content |
| `.d/` | LaTeX projects or grouped files |
| `#-` prefix | Chronological order (e.g., `1-`, `2.1-`, `2.2-`) |
| `*.1, *.2` | Parallel documents (same base number, `.1` = likely first) |

#### 5.2 Directory Structure
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

#### 5.3 README Requirements
- Add `README.md` to any directory with non-obvious organization
- Explain numbering schemes, content relationships, or special conventions

### 6. Safe Deletion
- Delete files only after verifying content is captured elsewhere
- Track deferred work in `TODO.md` with clear, actionable items
