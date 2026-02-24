# Week 05 Assignment Prompt (David Blei)

**Source:** User-provided Slack message.

**Text:**
@channel: apologies for the delay. our next meeting is:

friday 2/20 from 1p - 3p in hamilton 516

note the change in day, time, and room!

as for the topic:  last week we went over the premise and reasons for ICP. next week we'll get to an algorithm, specifically the probabilistic modeling approach of wu+ (2025). if time, i'm also interested in looking at other optimization based approaches.  please do reading around these circles of ideas.

bayesian ICP (our focus) : https://arxiv.org/abs/2506.22675
one approach to ICP via optimization : https://arxiv.org/abs/2412.11850

in addition to your reader report, i'd also like a project update. please write "what is the problem?", as before. then please write a more formal / mathematical definition of the problem. this might involve some setup (e.g. a causal graph or other assumptions). you may also need to invoke something about the strategy behind your solution. ideally, your problem is reduced to a handful of mathematical expressions (or just one). i'm expecting it to be <= 1 page, sometimes even just a few lines.

as an example, suppose i was working on ICP. the problem : what are the invariant variables across interventional environments of data? the mathematical formulation : each environment defines a joint p_e(x, y). we want to find variables s, such that p_e(y | x_s) = p(y | x_s) across all e.

**Attachments:**
None (papers linked by arXiv URL).

**Requirements Derived:**
1.  **Readings:**
    *   Wu et al. (2025) - "Bayesian Invariance Modeling of Multi-Environment Data" (arXiv:2506.22675) — **primary focus**
    *   Fan et al. (2024) - "Causal Invariance Learning via Efficient Nonconvex Optimization" (arXiv:2412.11850) — optimization-based approach
2.  **Deliverable:**
    *   Reader Report (Page 1: summary/critique of readings).
    *   Page 2: Project Update —
        *   "What is the problem?" (as before)
        *   **NEW:** Formal / mathematical definition of the problem (≤ 1 page, ideally a few lines).
            *   May involve causal graph, assumptions, strategy.
            *   Reduce to a handful of mathematical expressions.
3.  **Meeting Details:** Friday 2/20, 1p–3p, Hamilton 516 (changed day, time, room).
4.  **Themes:** Bayesian ICP, Probabilistic Modeling, Optimization-based ICP, NegDRO.
