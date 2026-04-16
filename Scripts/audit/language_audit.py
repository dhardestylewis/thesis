import fitz
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ── 1. Extract full thesis text ───────────────────────────────────────────────
pdf_path = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1\Austin_NIMBY_Thesis_Draft.pdf'
doc = fitz.open(pdf_path)
pages_text = [page.get_text() for page in doc]
full_text = '\n'.join(pages_text)

with open('thesis_full_text.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)
print(f"Extracted {len(full_text):,} characters across {len(doc)} pages")

# ── 2. JMLR / econometrics compliance patterns ───────────────────────────────

# A. Terms that JMLR papers AVOID (overclaiming, imprecision)
jmlr_avoid = [
    (r'\bsignificant(?:ly)?\b(?! difference| association| improvement)', 'JMLR: "significant" should only appear as statistical term — check usage'),
    (r'\bprove[sd]?\b', 'JMLR: use "show", "demonstrate", "establish" — not "prove"'),
    (r'\bdemonstrate[sd]? that\b.*\bcause[sd]?\b', 'CAUSAL: causal claim without identification strategy'),
    (r'\bour model\b', 'JMLR: use "the model" or specify model name (CatBoost)'),
    (r'\bstate.of.the.art\b', 'JMLR: avoid unless benchmarked vs. SOTA on same dataset'),
    (r'\bnovel\b', 'JMLR: "novel" is overused — be specific about what is new'),
    (r'\brobust(?:ly|ness)?\b', 'JMLR: "robust" is vague — specify to what perturbation'),
    (r'\bpowerful\b', 'JMLR: subjective — replace with measurable claim'),
    (r'\bshows that the model (?:is|can|will)\b', 'JMLR: overclaiming — include CI or hedge'),
    (r'\baccurat(?:e|ely|acy)\b(?! measure| label| reconstruction)', 'JMLR: "accurate" is vague — specify metric and value'),
]

# B. Economics / planning audience — causal discipline
econ_flags = [
    (r'\b(?:causes?|caused by|due to)\b', 'ECON: causal language — needs identification strategy or hedge'),
    (r'\beffect of\b', 'ECON: "effect" implies causation — use "association" or "relationship" without RD/IV'),
    (r'\bdriven by\b', 'ECON: "driven by" implies causation — hedge appropriately'),
    (r'\bexplain[s]?\b.*\b(?:variance|variation|outcome)\b', 'ECON: "explains variance" is R² language — distinguish from causal explanation'),
    (r'\bcalibrat(?:ion|ed|e)\b.*\b(?:performance|accuracy|well)\b', 'ECON/JMLR: "calibration" ≠ general performance — restrict to probability reliability sense'),
    (r'\bprecision\b(?!-recall| score| of)', 'ECON: check "precision" — is this PR precision or general accuracy? Stijn flagged this'),
]

# C. Register leakage — ML jargon in policy sections
ml_jargon_in_policy = [
    (r'\bhyperparamete\b', 'REGISTER: ML jargon — only in methods section'),
    (r'\bgradient boost(?:ing|ed)\b', 'REGISTER: spell out in non-technical sections'),
    (r'\bfeature importan\b', 'REGISTER: use "predictor weight" or "reliance" in policy-facing prose'),
    (r'\bSHAP\b', 'REGISTER: define on first use outside methods'),
    (r'\bOOD\b', 'REGISTER: spell out "out-of-distribution" outside methods'),
    (r'\bPR.AUC\b', 'REGISTER: define before using acronym'),
    (r'\bECE\b', 'REGISTER: define before using acronym'),
]

# D. Legacy patterns from PROSE_AUDIT.md that may have re-emerged
legacy_patterns = [
    (r'\bpipeline\b', 'LEGACY: "pipeline" — acceptable if generic, flag if over-used (>3x)'),
    (r'\bstructural\b', 'LEGACY: "structural" — is this causal/institutional/architectural? Be specific'),
    (r'\bregime\b', 'LEGACY: "regime" — ensure always means Austin regulatory regime, not metaphor'),
    (r'\boperationali[sz]e\b', 'LEGACY: "operationalize" — use sparingly, define on first use'),
    (r'\bfundamentally\b', 'LEGACY: "fundamentally" — intensifier, acceptable only in epistemic claims'),
    (r'\bframewor[k]\b', 'LEGACY: "framework" — vague, specify what kind'),
    (r'\binstitutional(?:ly|ize)?\b', 'LEGACY: "institutional" overuse — check count'),
]

# E. Missing epistemic markers (JMLR expects these)
missing_hedges = [
    (r'(?<!\blikely\b)(?<!\bappear\b)(?<!\bsuggest\b)(?<!\bseem\b)\bthe model (?:is|predicts|ranks)\b', 'JMLR: unhedged model claim — consider "the model appears to" or "the model tends to"'),
    (r'\bwe (?:see|find|observe)\b(?! that)', 'JMLR: "we see/find" without "that" — complete the hedge'),
    (r'\bconfirm[s]?\b', 'JMLR: "confirms" overclaims — use "is consistent with"'),
    (r'\bvalidat(?:es?|ing|ion)\b(?! set| split| procedure| check| layer| protocol)', 'JMLR: "validates" overclaims — use "is consistent with" or "supports"'),
]

# ── 3. Run analysis ──────────────────────────────────────────────────────────

all_flags = defaultdict(list)
categories = {
    'JMLR Compliance': jmlr_avoid,
    'Econometrics Discipline': econ_flags,
    'Register Leakage (ML→Policy)': ml_jargon_in_policy,
    'Legacy Jargon (from prior audit)': legacy_patterns,
    'Missing Epistemic Hedges': missing_hedges,
}

# Work sentence by sentence
sentences = re.split(r'(?<=[.!?])\s+', full_text)

for cat_name, patterns in categories.items():
    for pattern, msg in patterns:
        matches = []
        for page_num, page_text in enumerate(pages_text, 1):
            for m in re.finditer(pattern, page_text, re.IGNORECASE):
                # Get surrounding context
                start = max(0, m.start() - 80)
                end = min(len(page_text), m.end() + 80)
                context = page_text[start:end].replace('\n', ' ').strip()
                matches.append((page_num, m.group(), context))
        if matches:
            all_flags[cat_name].append((msg, matches))

# ── 4. Word frequency for "structural" and high-frequency suspect terms ──────
term_counts = {}
suspect_terms = ['structural', 'regime', 'pipeline', 'institutional', 'operationalize',
                 'fundamentally', 'framework', 'significant', 'robust', 'novel',
                 'calibration', 'precision', 'accurate', 'demonstrate', 'validate']
for term in suspect_terms:
    count = len(re.findall(r'\b' + term + r'\w*\b', full_text, re.IGNORECASE))
    term_counts[term] = count

# ── 5. Print report ──────────────────────────────────────────────────────────
print('\n' + '='*70)
print('TERM FREQUENCY AUDIT')
print('='*70)
for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
    flag = ' ⚠️ HIGH' if count > 15 else (' 👀' if count > 8 else '')
    print(f"  {term:<25} {count:>4}x{flag}")

print('\n' + '='*70)
print('FLAGGED PATTERNS BY CATEGORY')
print('='*70)
for cat_name, flags in all_flags.items():
    print(f"\n### {cat_name} ###")
    for msg, matches in flags:
        print(f"\n  [{len(matches)}x] {msg}")
        for page_num, matched_text, context in matches[:3]:  # show up to 3 examples
            print(f"    p.{page_num}: ...{context}...")
        if len(matches) > 3:
            print(f"    ... and {len(matches)-3} more occurrences")

print('\n' + '='*70)
print(f"Total flag categories: {sum(len(v) for v in all_flags.values())}")
