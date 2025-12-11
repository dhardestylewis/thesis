# Thesis Project Guidelines

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
- Append conversation prompts to `Prompts-COMPREHENSIVE.txt`
- Record session changes in `CHANGELOG.md` with date and time

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
| `.d/` | LaTeX projects with auxiliary files |

#### 5.2 Directory Structure
```
thesis/
├── Thesis_Draft/           # Active working files
├── Assignments_and_Proposal-SUBMITTED/  # Finalized coursework
├── Deprecated_Writings/    # Archived content
├── TODO.md                 # Prioritized task list
├── GUIDELINES.md           # This file
└── CHANGELOG.md            # Session history
```

### 6. Safe Deletion
- Delete files only after verifying content is captured elsewhere
- Track deferred work in `TODO.md` with clear, actionable items
