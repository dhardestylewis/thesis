# Thesis Project Guidelines

## Fact Verification Rules

1. **Always externally verify factual claims** before marking as correct
   - Use web search to confirm dates, names, percentages, and events
   - Compare document claims against authoritative sources (government sites, reputable news)

2. **Prioritize authoritative sources for citations**:
   - Government sources (texas.gov, austintexas.gov, capitol.texas.gov)
   - Academic sources (university research)
   - Established news organizations (Texas Tribune, KUT)
   - Official organizational sites

3. **Check that citations actually exist in references.bib**
   - Verify inline `\cite{key}` commands have matching bib entries
   - Search using multiple keywords (author name, title words, year)

### Verification Approach (step-by-step)
1. Search for claims in target document
2. Check if cited reference exists in references.bib
3. Externally verify factual accuracy via web search
4. Compare authoritative sources found vs current citations
5. Update bib entries or inline citations as needed

## Bibliography Management Rules

4. **Use consistent BibTeX formatting**:
   - Blank line after `@type{key,`
   - Blank line between each field
   - Two-space indentation for fields
   - Blank line before closing `}`

5. **Organize bibliography with commented section headers**:
   - Primary Sources (Government, Court)
   - Secondary Sources (Academic)
   - News Sources
   - Organizational Sources
   - Data Sources

6. **When adding missing entries**, use verified external sources and note the verification date

## File Organization Rules

7. **Naming conventions**:
   - Use `-SUBMITTED` suffix for finalized/submitted work
   - Use `-TO_INTEGRATE` suffix for content pending integration
   - Use `-COMPREHENSIVE` for complete/long-form documents
   - Use `.d/` directories for LaTeX projects with auxiliary files

8. **Directory structure**:
   - `Thesis_Draft/` — Active working files for thesis
   - `Assignments_and_Proposal-SUBMITTED/` — Finalized coursework
   - `Deprecated_Writings/` — Archived/superseded content

9. **Track deferred work in `TODO.md`** with clear, actionable items

## Documentation Rules

10. **Append conversation prompts** to `Prompts-COMPREHENSIVE.txt` for continuity
11. **Record session changes in `CHANGELOG.md`** with date and time
12. **Commit frequently** with descriptive messages summarizing changes
13. **Delete files only after verifying** their content is captured elsewhere or no longer needed
