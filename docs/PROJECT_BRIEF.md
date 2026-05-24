# Project: Multilingual Cultural-Sensitivity & Emotional-Appropriateness Evaluation Pipeline for Conversational AI

## 1. Background and motivation

Conversational AI assistants serving an international user base sometimes produce responses that lack emotional appropriateness or fail to respect cultural norms — wrong formality level, stereotyping, tone-deaf replies to emotional cues, or mishandling of culturally specific topics. These qualities are subjective and hard to measure, yet a team needs reliable, scalable metrics to decide whether a feature is good enough to launch and to identify what to improve.

This project is a small, self-contained prototype that demonstrates a credible methodology for measuring those fuzzy qualities. It is intentionally built on synthetic, self-authored data: the value is in the evaluation methodology and its rigor, not the dataset size. The same pipeline would scale to real labeled data and native-speaker annotators.

The single most important idea the project must demonstrate: taking a vague quality ("cultural sensitivity") and operationalizing it into concrete, measurable sub-criteria, then building an LLM-as-Judge to score against those criteria *and validating that judge against human labels* — because an unvalidated judge cannot be trusted to drive decisions.

## 2. Objectives

The project should, by the end, demonstrate the ability to: operationalize fuzzy constructs (cultural sensitivity, emotional appropriateness) into a concrete scoring rubric with definitions and examples; generate a small evaluation dataset of conversational prompts and AI responses spanning multiple locales; implement an LLM-as-Judge that scores responses against the rubric; validate the judge by measuring its agreement with hand-created human ground-truth labels; surface and mitigate known judge biases (position bias, verbosity bias, self-enhancement/model-family bias, etc.); and analyze the results, including a defect distribution showing where failures cluster.

## 3. Scope and explicit non-goals

In scope: a small synthetic dataset (target 70-100 prompts), 2–3 locales/languages, a documented rubric, an LLM-as-Judge implementation, a human-labeled validation subset (20-30 items), agreement analysis, bias checks, a defect-distribution analysis, and a clear written report.

Explicitly out of scope, and the README should say so: real or scraped user data of any kind (everything is synthetic and self-authored — this is a hard requirement, never use anything resembling real assistant logs); training or fine-tuning a model; large-scale data collection; production engineering, deployment, or a UI; native-speaker-grade annotation (you are a stand-in "annotator," and the writeup should acknowledge that real evaluation needs native speakers per locale).

## 4. Key design decisions to make early

A few choices shape everything else, so decide them at the start. Pick the locales — a good combination is one high-formality/honorific-heavy language (e.g. Japanese), one where the scientist can reason about cultural specifics confidently (e.g. Bangla), and English as a baseline; and note in the report where you lack native fluency (e.g. Japanese). Decide the scoring mode: use **pairwise comparison** (judge picks the better of two responses) as the primary mode because relative judgments are more reliable than absolute scores, and optionally include pointwise 1–5 scoring on each sub-criterion for richer per-criterion analysis. First, design the evaluation pipeline for pairwise comparison. Next iteration, use pointwise 1–5 scoring and compare outcomes of each approaches.  Decide how to generate the "responses" to be judged — the cleanest approach is to author, for each prompt, one deliberately good response and one deliberately flawed response (flawed in a specific, labeled way such as "too informal for locale" or "ignores user's distress"), which gives you a known ground truth to validate the judge against. Decide which model plays the judge and keep it fixed across runs for consistency.

## 5. Proposed methodology and pipeline

The flow is: define the rubric, then build the dataset of prompts each paired with a good and a flawed response (with the flaw type recorded), then create your own human ground-truth labels for the validation subset before looking at any judge output (to avoid anchoring), then run the LLM-as-Judge (with structed output in json) in pairwise mode with each pair evaluated in both orderings to detect position bias, then compute judge-vs-human agreement on the validation subset, then run the judge across the full set and produce the defect distribution and summary metrics, then write up findings, limitations, and how this connects to model training.

The rubric is the intellectual core. Decompose cultural sensitivity into observable sub-criteria — appropriate formality/register and honorifics for the locale; avoidance of stereotypes and overgeneralizations; correct, respectful handling of culturally specific topics (holidays, food, customs, names); absence of inappropriate cultural assumptions. Decompose emotional appropriateness into: acknowledges the user's emotional state when present; tone matches the situation (empathy for distress, warmth for good news); avoids tone-deaf or dismissive replies. Each sub-criterion needs a one-line definition and a short positive and negative example. This rubric should live in its own document and be treated as a deliverable in its own right.

For validation, the principle to honor is that you never trust the judge blind. You create human labels first, independently, then measure agreement (report raw agreement rate and Cohen's kappa for the pairwise decisions). Discuss specific disagreements and what they reveal — genuine ambiguity vs. a vague rubric criterion vs. a judge limitation. For bias, run every pairwise comparison twice with the two responses in swapped positions; if the judge's preference flips with order, that's position bias — report how often it happens and mitigate by averaging/ignoring inconsistent pairs.

## 6. Connecting back to training (include in the report)

Close the loop explicitly in the writeup: the pairwise preference judgments this pipeline produces are exactly the form of preference data that feeds modern alignment methods (RLHF, DPO). So the same artifact that measures quality also generates training signal, and a validated LLM-as-Judge can scale that signal cheaply once trusted. One short paragraph stating this demonstrates you see the full eval-to-training flywheel rather than evaluation in isolation.

## 7. Deliverables

A short report/README (one to two pages) stating the problem, approach, key findings, limitations, and "what I'd do next with real data and native-speaker annotators." The rubric document. The synthetic dataset (prompts + paired responses + flaw labels). The judge implementation and the analysis code/notebook. An agreement-and-bias analysis with numbers. A defect-distribution summary (where do failures cluster — by locale? by criterion? by flaw type?). The report is the most important deliverable; lead with the narrative, not the code.

## 8. Suggested tech stack and structure

Python in a notebook or small scripts; an LLM API for the judge and optionally for response generation; pandas for analysis; a simple agreement/kappa calculation (scikit-learn provides Cohen's kappa). A clean repo layout: a `README.md` (the report), a `rubric.md`, a `data/` folder for the synthetic dataset, a `src/` or notebook for the pipeline, and a `results/` folder for outputs and the defect-distribution summary.

## 9. Important guardrails

All data must be synthetic and self-authored — never use, scrape, or imitate real assistant logs or user data. Keep the dataset small and the methodology rigorous rather than chasing scale. Be honest in the writeup about every limitation (small n, single non-native annotator, synthetic data, one judge model). State assumptions explicitly. The maturity shown by a strong "limitations" section is itself part of what impresses.

## 10. Task list (suggested execution order)

1. Set up the repo structure and write a one-paragraph problem statement in the README.
2. Choose the 3 locales and the judge model; record these decisions and their rationale in the README.
3. Draft `rubric.md`: break cultural sensitivity and emotional appropriateness into sub-criteria, each with a definition and a positive/negative example.
4. Author the synthetic prompts (aim 80-100), spread across the chosen locales and across both cultural and emotional scenarios. The distribution across locales should not be a balanced distribution, such as: English 50%, Bangla 35%, Japanese 15%.
5. For each prompt, author one good response and one deliberately flawed response, recording the flaw type. Store as structured data (e.g. CSV/JSON).
6. Create human ground-truth pairwise labels for a 20-30 item validation subset, and ask the scientist to label it, before running the judge.
7. Implement the LLM-as-Judge: pairwise mode, scoring against the rubric, returning a decision plus brief rationale.
8. Run the judge on the validation subset in both position orderings; record results.
9. Compute judge-vs-human agreement (raw agreement + Cohen's kappa) and quantify position bias (flip rate); write up notable disagreements.
10. Run the judge across the full dataset; produce the defect distribution (failures by locale, by sub-criterion, by flaw type).
11. (Optional) Add pointwise 1–5 per-criterion scoring for a richer breakdown.
12. Write the report: problem, approach, findings, agreement/bias results, defect distribution, limitations, and the connection to model training (RLHF/DPO).
13. Final pass: verify all data is synthetic, assumptions are stated, and the README reads as a clear narrative a busy manager can skim.
