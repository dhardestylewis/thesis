# Rethinking the Austin Zoning Thesis: Four Linked Studies

Here is the version I would actually run, front to back.

## 0. Freeze the study before modeling

Write a short preregistration that locks five things before fitting any serious model: the **primary predictive target**, the **primary causal estimands**, the **forecast horizons**, the **test folds**, and the **primary metrics**. The current draft feels like the target moves between protest, petition, dissenting votes, and outcomes; the next iteration needs one primary target and then clearly labeled secondary tasks. 

I would preregister these:

* **Primary predictive question:** At the time a case is filed, what is the probability that it will face **organized opposition** before final council action?
* **Primary causal question 1:** What is the effect of crossing the **valid protest threshold** on council dissent and approval behavior in the pre-H.B. 24 regime?
* **Primary causal question 2:** How did **HOME Phase 1, HOME Phase 2, and H.B. 24** change opposition and voting dynamics for legally affected cases?
* **Governance question:** Can this model be used for **proactive outreach** without becoming a developer targeting tool?

## 1. Build one clean data warehouse with time-stamped “as-of” snapshots

The warehouse should be built from official sources and every record should carry an **as-of date** so you can reconstruct what was knowable at each prediction point. The obvious backbone is Austin’s official Zoning Cases data, plus zoning-review GIS data, TCAD public appraisal/EARS extracts, ACS 5-year tract and block-group releases, and macro series from FRED/ALFRED. Using as-of joins matters because the Census and macro sources are revised and updated over time; real backtests should only use the latest release available at the forecast date. ([City of Austin Open Data Portal][2])

I would create these tables:

1. **case_master**

   * case_id
   * filing date, notice date, petition deadline, planning commission date, council date, withdrawal date
   * requested zoning from/to
   * requested units, FAR, height, lot size, use changes
   * status and disposition

2. **site_geometry**

   * parcel ids for subject site
   * polygon, acreage, frontage, corner-lot flag
   * 200 ft / 500 ft / 1000 ft buffers
   * census tract / block group / council district / neighborhood planning area

3. **parcel_buffer_snapshot**

   * year-specific TCAD variables for parcels in each buffer
   * land value, improvement value, land-to-total ratio
   * homestead exemption share
   * owner-occupancy share
   * structure age, parcel size, improvement size

4. **neighborhood_snapshot**

   * ACS variables by tract/block group
   * tenure, income, rent burden, vacancy, race/ethnicity composition, education, household type, car ownership, limited-English share
   * displacement-vulnerability overlays

5. **policy_calendar**

   * 2022 council regime change
   * HOME Phase 1 adoption and application-acceptance start
   * HOME Phase 2 adoption and application-acceptance start
   * H.B. 24 effective date and eligibility flags derived from the final statute

6. **meeting_event**

   * planning commission recommendation
   * council agenda placement
   * staff recommendation
   * continuances, postponements

7. **speech_comment**

   * speaker id if available
   * stance: support / oppose / neutral
   * text, timestamp, meeting type
   * text segments for later frame coding

8. **petition_record**

   * petition filed y/n
   * validity y/n
   * raw signed-area share if recoverable
   * signatory count
   * petition text

9. **vote_record**

   * case × councilmember
   * yes / no / abstain / absent / recused
   * district relationship and timing

10. **audit_log**

* missingness
* geocode match quality
* transcript alignment quality
* document extraction QA

## 2. Put the whole design on a process timeline so leakage is impossible

This is the single biggest structural fix. The draft uses hearing transcripts as predictive signal, but those transcripts belong late in the process. The next iteration should score **different horizons separately** instead of pretending one model serves all time points. 

I would define four forecast horizons:

* **H0: filing model**

  * everything known at filing only
  * this is the true ex ante model

* **H1: notice / petition-deadline model**

  * H0 plus notices, early written comments, petition text if filed

* **H2: pre–planning commission model**

  * H1 plus staff report and planning commission packet

* **H3: pre-council model**

  * H2 plus planning commission outcome and planning-commission testimony
  * council-hearing testimony is allowed only here if the task is “before final vote,” not “at filing”

That gives you a clean answer to the policy question: what can be predicted **when**?

## 3. Define one primary outcome, then keep the rest as secondary tasks

I would make the **primary predictive outcome**:

**Organized opposition before final council action** = 1 if any of the following occurs:

* a valid protest petition is filed, or
* at least 3 distinct opposing speakers/comments are recorded before final council action, or
* opposition comments exceed support comments by a prespecified margin.

Why this composite? Because it captures mobilized opposition better than petition-only and is still more concrete than “NIMBYism.” Then I would vary the threshold in robustness checks.

Secondary outcomes:

* **y1:** valid petition filed
* **y2:** opposition speaker/comment count
* **y3:** opposition severity ordinal score (0 none, 1 low, 2 moderate, 3 high)
* **y4:** count of dissenting council votes
* **y5:** approval / denial / continuance
* **y6:** councilmember-level no-vote indicator

Also add a **censoring rule**:

* cases withdrawn before public notice are **not negatives**
* cases with missing petition files or missing transcript coverage are **unknown**, not negatives
* cases with administrative closures are excluded from the primary predictive sample

This matters because meeting participation is not representative; local land-use participation skews toward older, male, longtime residents, voters, and homeowners, and strongly against new housing. That means observed opposition is not the same thing as latent neighborhood preference. ([Maxwell Palmer][3])

## 4. Feature set: two versions, one for research and one for deployment

I would build two feature libraries.

### A. Deployment-eligible feature set

Only variables available at the forecast date and acceptable for public-sector use.

**Case and site**

* zoning from / zoning to
* density change magnitude
* height change
* FAR change
* requested unit count change
* requested lot size change
* PUD / TOD / VMU / corridor / overlay flags
* subject-site acreage
* frontage
* corner-lot flag
* adjacency to arterial / transit corridor
* floodplain / environmental constraint flags

**Legal / procedural**

* petition-eligible under pre-H.B. 24 rules
* H.B. 24-eligible under final statute
* HOME Phase 1 eligibility
* HOME Phase 2 eligibility
* owner-initiated vs city-initiated
* comprehensive-zoning-change flag if relevant under statute
* election-cycle timing
* district of site
* whether case is in district of current councilmember

**Parcel-buffer economics (200 ft, 500 ft, 1000 ft)**

* median appraised value
* mean land value
* mean improvement value
* median land-to-total ratio
* homestead-exemption share
* owner-occupancy share
* median structure age
* share single-family vs multifamily parcels
* parcel size distribution
* assessed-value growth over past 1, 3, 5 years

**Neighborhood demographics**

* renter share
* median household income
* rent burden
* vacancy rate
* family-with-children share
* car ownership
* education
* population density
* recent permit volume / demolitions / rezonings nearby

**Political history**

* prior opposition rate within 500 m / tract / district
* prior valid-petition rate nearby
* district-level dissent history
* whether similar recent cases nearby produced conflict

**Macro / market**

* mortgage rate
* fed funds rate
* local vacancy proxy
* local rent-growth proxy
* local home-price-growth proxy

### B. Research-only / audit-only feature set

These are useful for mechanism and fairness auditing, but I would **not** put them in the deployment model by default.

* tract racial/ethnic composition
* segregation overlays
* displacement-vulnerability overlays
* historic exclusion / east-west geography proxies
* any personally identifying owner names
* any inferred race-from-name feature

That split is important because disparate impact can emerge from apparently neutral data, and public-sector tools should be especially conservative about sensitive or proxy variables. ([Yale Computer Science][4])

## 5. Add a real text pipeline, but only where the timing allows it

I would not throw generic sentence embeddings into one big model and call that mechanism. I would build text in two layers:

### Layer 1: stance classification

For each comment, petition paragraph, or testimony segment:

* support
* oppose
* neutral / procedural

### Layer 2: frame classification

Multi-label coding for:

* traffic / parking
* infrastructure capacity
* schools / services
* flood / environmental concerns
* displacement / gentrification
* affordability / pro-housing
* property values / taxes
* neighborhood character / compatibility
* project execution / design specifics
* procedural fairness / notice
* safety / crime
* legal / process arguments

I would hand-label a stratified seed set:

* 2,000 speech segments
* 500 petition paragraphs
* 500 written comments

Use two coders plus adjudication. Metrics:

* Krippendorff’s alpha for each label
* micro-F1
* macro-F1
* per-label AUPRC
* exact stance accuracy

Then use active learning to add hard examples from rare regimes or project types.

Most important: these text outputs should be interpreted as **descriptive rhetorical signals**, not as direct evidence of the “true causal mechanism.” That is exactly where the current draft overreaches. 

## 6. Use splits that mimic the real failure modes

The current draft uses many temporal/environment partitions, but the next iteration needs a smaller number of **substantive** environments tied to real policy regimes. ICP-style methods only make causal claims under invariance assumptions across environments, and V-REx is only meaningful if the environment definition is itself substantively motivated. ([ETH Zurich Mathematics Homepages][5])

I would use these outer test designs:

### A. Rolling-origin temporal test

For test year (t):

* train on years < (t-1)
* validate on year (t-1)
* test on year (t)

### B. Policy-regime holdouts

* pre-2022 council
* 2022 transition
* HOME Phase 1 application window starting **Feb. 5, 2024**
* HOME Phase 2 partial application window starting **Aug. 16, 2024**
* HOME Phase 2 citywide window starting **Nov. 16, 2024**
* post–H.B. 24 starting **Sept. 1, 2025**. ([City of Austin][1])

### C. Spatial holdout

Leave out one council district or one planning area at a time.

### D. Site-family holdout

Group all amendments / continuances / resubmissions for the same site into one fold.

### E. Actor holdout

Hold out developers / applicants / owners that appear often, to test memorization.

### F. Project-type holdout

PUDs, neighborhood plan amendments, missing-middle / HOME-type cases, commercial rezonings.

### G. Worst-group/OOD evaluation

For every chosen model, report not just average performance but:

* worst-regime PR-AUC
* worst-regime Brier score
* worst-regime calibration slope
* max drop from in-distribution to OOD fold

## 7. Prediction model stack: start simple, then earn complexity

Because your outcome is imbalanced, **PR-AUC must be the lead discrimination metric**, not ROC-AUC. ROC can stay high even when the model is almost useless in the positive class. The draft itself already shows that pattern.  ([PLOS][6])

I would fit models in this order:

### Baselines

1. prevalence by year/regime
2. elastic-net logistic regression
3. hierarchical logistic regression with district random effects

### Strong tabular models

4. CatBoost or LightGBM on structured features
5. structured-only model at H0
6. structured + text late-fusion at H1/H2/H3

### Robust/OOD models

7. V-REx on top of the best structured or fusion model
8. anchor-regression-style linear benchmark for interpretability
9. Bayesian invariant prediction as a robustness check, not the centerpiece

### What I would not make central

10. diffusion augmentation
11. CVAE/DDPM as the main predictive engine

Diffusion is the kind of thing I would test only **after** simpler imbalance remedies fail, and only if synthetic cases pass realism, privacy, and downstream-calibration audits.

For class imbalance I would try, in order:

* class-weighted loss
* focal loss
* balanced minibatches
* threshold tuning
* only then synthetic augmentation

Hyperparameter search I would keep boring and nested:

* elastic net: (C \in [10^{-4}, 10^2]), (l1_ratio \in {0, .25, .5, .75, 1})
* boosted trees: depth 4–8, learning rate 0.02–0.1, trees 300–3000, subsample 0.6–1.0
* fusion MLP: hidden sizes 256→128, dropout 0.1–0.3
* text fine-tuning: lr (1e^{-5}) to (3e^{-5}), 2–4 epochs
* V-REx: (\lambda \in {0, .1, 1, 10, 100})

Model selection rule:

1. maximize validation PR-AUC
2. subject to calibration slope between 0.9 and 1.1
3. tie-break with lowest worst-regime log loss

For uncertainty, I would add:

* cluster/block bootstrap CIs
* held-out calibration window
* conformal-style uncertainty or abstention bands for cases far from the training support. ([arXiv][7])

## 8. Metrics: this should be exhaustive and prespecified

### Data-quality metrics

* geocode match rate
* parcel-join completeness
* transcript-to-case alignment precision/recall
* petition extraction field-level F1
* missingness by year/regime
* duplicate-case rate
* manual adjudication agreement

### Prediction metrics

**Primary**

* PR-AUC overall
* PR-AUC by horizon
* worst-regime PR-AUC

**Secondary discrimination**

* ROC-AUC
* precision@top 5%, 10%, 20%
* recall@top 5%, 10%, 20%
* lift@top 5%, 10%, 20%
* balanced accuracy
* MCC
* F1 at a prespecified operational threshold

**Probabilistic quality**

* log loss
* Brier score
* calibration intercept
* calibration slope
* reliability diagrams
* ECE
* local calibration error

Use ECE only as a summary check, not the whole calibration story, because average calibration metrics can hide finer-grained failures; local calibration helps surface those. ([Journal of Machine Learning Research][8])

### OOD/stability metrics

* in-distribution vs OOD PR-AUC gap
* in-distribution vs OOD Brier gap
* worst-environment loss
* variance of environment losses
* rank stability of top features across folds
* drift score by regime/year

### Text metrics

* stance accuracy
* stance macro-F1
* frame micro-F1
* frame macro-F1
* per-label AUPRC
* Krippendorff’s alpha / Cohen’s kappa

### Fairness / governance metrics

Report all by:

* council district
* east/west geography
* majority-renter vs majority-owner tracts
* majority-Black / Hispanic / White tracts
* high vs low displacement-risk areas

Metrics:

* calibration within group
* false positive rate gap
* false negative rate gap
* equal opportunity gap
* precision gap at operational threshold

And explicitly note the tradeoff: in general you cannot simultaneously satisfy group calibration and equalized-odds-style error parity except in trivial cases, so this needs to be reported as a design tradeoff, not hidden. ([NeurIPS Papers][9])

## 9. Causal study 1: replace the OLS with a threshold design if possible

The cleanest causal study in your whole setting is probably **not** DiD. It is a threshold design around the old petition-validity rule.

The current draft already implies a binary “valid petition” indicator tied to a specific statutory threshold. If you can recover the **continuous signed-area share** behind that binary, I would run a local regression discontinuity or fuzzy RD around the pre-H.B. 24 cutoff. That is far stronger than OLS of no-vote counts on a petition dummy. 

Running variable:

* signed-area share within the statutory protest buffer

Treatment:

* valid petition status

Outcomes:

* councilmember no-vote
* dissenting-vote count
* continuance / postponement
* approval / denial

Diagnostics:

* bandwidth sensitivity
* covariate continuity
* density test around cutoff
* placebo cutoffs
* donut RD excluding heaped edge cases

If the continuous running variable is not recoverable, then I would fall back to:

* case × councilmember panel
* doubly robust / DML estimation with cross-fitting
* heterogeneous effects via causal forests

That gives you a credible average effect plus principled heterogeneity exploration without pretending simple associations are causal. ([MIT Economics][10])

## 10. Causal study 2: HOME should be two separate event studies

HOME Phase 1 and HOME Phase 2 should be estimated as **different treatments**, because they had different adoption and application dates and different legal consequences. Use application-acceptance dates, not just adoption dates, because treatment starts when cases can actually be filed under the new rules. ([City of Austin][1])

I would define treatment at the **case level** based on parcel eligibility:

* newly eligible under HOME Phase 1
* newly eligible under HOME Phase 2
* ineligible comparison cases

Outcomes:

* filing volume
* organized opposition probability
* opposition intensity
* dissenting votes
* approval time / continuance

Estimator:

* Callaway–Sant’Anna group-time ATT
* Sun–Abraham style dynamic event study for leads/lags

Do **not** use plain TWFE with leads/lags as the main specification, because heterogeneous treatment timing can contaminate those coefficients. ([ScienceDirect][11])

Diagnostics:

* pretrends
* alternate control groups
* placebo implementation dates
* tract-specific linear trends
* sensitivity to excluding transition months

## 11. Causal study 3: H.B. 24 should be coded from the statute, not a post dummy

I would not model H.B. 24 as “post-Sept-2025” and stop there. The final statute changed protest procedures with specific scope and effective date, so the treatment needs to be derived from **actual legal eligibility**, not assumed from calendar time alone. ([Texas Legislature Online][12])

Build an explicit case-level flag:

* owner-requested rezoning?
* type of zoning change?
* comprehensive vs non-comprehensive status if relevant
* residential vs commercial effect under final law
* grandfathered or not based on application timing

Then estimate:

* eligible vs ineligible cases
* before vs after Sept. 1, 2025
* dynamic treatment effects with modern staggered DiD estimators
* plus placebo dates and placebo unaffected outcomes

That is a much better test than “residential × post.”

## 12. Interpretability: descriptive, stable, and honest

Keep interpretability, but change its role.

I would report:

* SHAP or permutation importance for predictive attribution
* partial dependence / ALE plots
* fold-to-fold stability of signs and ranks
* feature-group ablations

I would **not** describe SHAP shifts as proof of “true causal mechanisms.” ICP can be informative under strong assumptions, V-REx can improve robustness under distribution shift, and Bayesian invariant prediction can quantify uncertainty over invariant sets, but none of these by themselves license the causal rhetoric in the current draft.  ([ETH Zurich Mathematics Homepages][5])

The ablation order I would publish is:

1. structured baseline
2. * legal/policy features
3. * parcel-buffer features
4. * neighborhood history
5. * macro features
6. * text features allowed at that horizon
7. * OOD/invariance regularization
8. * synthetic augmentation

That lets the reader see what actually buys signal.

## 13. Qualitative work should move from “illustration” to governance experiment

The qualitative component should stop trying to validate quantitative causal claims and instead become a **governance and misuse study**. The current draft’s strongest ethical point is the risk of developer targeting; that should become a formal red-team exercise, not just a cautionary paragraph. 

I would sample roughly:

* 8 city planners / zoning staff
* 8 anti-displacement / housing-equity staff
* 8 tenant or neighborhood organizers
* 8 affordable-housing developers
* 8 market-rate developers
* 8 elected-office or commission stakeholders

Then run scenario-based interviews:

* risk score shown vs not shown
* parcel-level map vs neighborhood-decile summary
* outreach framing vs investor framing
* explanation panel vs black-box score

Qualitative outcomes:

* perceived legitimacy
* expected misuse
* outreach utility
* trust
* willingness to use internally
* willingness to release publicly

Coding themes:

* fairness
* targeting risk
* displacement concern
* interpretability
* procedural legitimacy
* human override
* accountability

## 14. Deployment rules should be strict

Because the current draft itself flags developer targeting as a core risk, the default should be:

* **no public parcel-level low-opposition map**
* internal access only
* use only for **extra outreach allocation**
* never use for denial, delay, or disadvantage of applicants
* every prediction accompanied by uncertainty and top contributing factors
* automatic abstain if case is OOD
* all access logged and audited

I would set explicit release gates:

* H0 model must beat elastic-net baseline by at least **25% relative PR-AUC**
* top-decile lift must exceed **2.0**
* calibration slope must be **0.9–1.1** overall and **0.8–1.2** in each audit group
* worst-group PR-AUC must be at least **80%** of overall PR-AUC
* no false-negative-rate gap above **10 percentage points** across audit groups
* if “low predicted opposition” correlates too strongly with displacement-vulnerability tracts, do not release beyond a tightly controlled internal pilot

That last rule is crucial. A model that is good at identifying low-resistance sites may also be good at identifying politically weak neighborhoods. That is exactly the harm you want to block.  ([Yale Computer Science][4])

## 15. What I would drop from the current version

I would explicitly cut five things:

* **same-hearing transcript embeddings** from any model that claims to be ex ante
* **OLS on dissent-vote count** as the main causal design
* **simple residential × post DiD** as the main policy design
* **diffusion augmentation as a headline contribution**
* **SHAP/ICP rhetoric that implies causal validation**

Those are the main reasons the current draft feels too ambitious for the evidence it has. 

## The short version of the redesign

The next iteration should be:

1. **One clean primary target:** organized opposition.
2. **Four forecast horizons:** filing, notice, pre-commission, pre-council.
3. **One disciplined prediction stack:** elastic net → boosted trees → late-fusion text → V-REx/BIP robustness.
4. **One strong causal design for petitions:** threshold/RD if possible, DML if not.
5. **Two separate policy studies:** HOME Phase 1/2 and H.B. 24, with modern event-study estimators.
6. **One governance study:** not “is the model clever,” but “can it be used without targeting vulnerable neighborhoods.”

The clean dissertation version is: **predict early opposition, estimate the petition effect with a threshold-based design, estimate HOME/H.B. 24 with modern event studies, and then audit the tool for misuse before anyone sees a map.**

[1]: https://www.austintexas.gov/development-services/home-amendments "https://www.austintexas.gov/development-services/home-amendments"
[2]: https://data.austintexas.gov/Building-and-Development/Zoning-Cases/edir-dcnf "https://data.austintexas.gov/Building-and-Development/Zoning-Cases/edir-dcnf"
[3]: https://maxwellpalmer.com/docs/articles/Einstein_Glick_Palmer_Participation.pdf "https://maxwellpalmer.com/docs/articles/Einstein_Glick_Palmer_Participation.pdf"
[4]: https://www.cs.yale.edu/homes/jf/BarocasSelbst.pdf "https://www.cs.yale.edu/homes/jf/BarocasSelbst.pdf"
[5]: https://people.math.ethz.ch/~peterbu/Files/Manuscripts/invariant-causal-prediction.pdf?utm_source=chatgpt.com "Causal inference by using invariant prediction"
[6]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432&utm_source=chatgpt.com "The Precision-Recall Plot Is More Informative than the ROC ..."
[7]: https://arxiv.org/abs/2107.07511?utm_source=chatgpt.com "[2107.07511] A Gentle Introduction to Conformal Prediction ..."
[8]: https://www.jmlr.org/papers/volume23/22-0658/22-0658.pdf?utm_source=chatgpt.com "Metrics of Calibration for Probabilistic Predictions"
[9]: https://papers.neurips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf "https://papers.neurips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf"
[10]: https://economics.mit.edu/sites/default/files/2022-08/2017.06%20Double%20Debiased%20Machine%20Learning%20for%20Treat.pdf "https://economics.mit.edu/sites/default/files/2022-08/2017.06%20Double%20Debiased%20Machine%20Learning%20for%20Treat.pdf"
[11]: https://www.sciencedirect.com/science/article/pii/S0304407620303948?utm_source=chatgpt.com "Difference-in-Differences with multiple time periods"
[12]: https://capitol.texas.gov/tlodocs/89R/billtext/html/HB00024F.htm "https://capitol.texas.gov/tlodocs/89R/billtext/html/HB00024F.htm"
