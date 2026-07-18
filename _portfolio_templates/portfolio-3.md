---
title: "Substantiveness Classification for Open-End Survey Responses"
excerpt: "Built a two-stage LLM classification pipeline that evaluates the depth and quality of open-ended survey responses, categorizing them into four tiers of substantiveness. The system integrates with an existing qualitative analysis infrastructure and includes schema migrations, async task orchestration, Jinja2 prompt templates, and a multiprocessing batch runner.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

## Project Overview

Swayable is a research platform that measures persuasion through online surveys. Surveys frequently include open-ended questions that invite respondents to share opinions in their own words. Historically, the platform classified these responses for relevance and sentiment — but lacked any measure of *depth*. A response that says "I liked it" and one that says "I found the pacing compelling because it reminded me of how long-form journalism creates emotional investment" were treated identically, despite their vastly different analytical value.

This ticket, ENG-1637, introduced **Substantiveness Classification**: an automated system that evaluates the intellectual depth of survey responses and assigns each one to one of four categories — `highlySubstantive`, `moderatelySubstantive`, `minimallySubstantive`, or `insubstantive`. The classifier targets opinion-type questions specifically, distinguishing them from recall, entity-listing, or attribute-listing question types. The expected outcome was a new dimension of qualitative data that would allow researchers to filter for high-signal responses, surface the most analytically useful answers at scale, and give clients a richer picture of audience engagement with their content.

The project required end-to-end implementation: database schema changes, LLM prompt engineering, a new Celery task layer, a REST API endpoint, and a command-line batch runner for backfilling and testing. It was completed and merged on February 9, 2026.

## Technical Approach

### Two-Stage Classification Architecture

The most important design decision in this project was introducing a prerequisite classification step before substantiveness scoring. Substantiveness is only meaningful for opinion-type responses — asking whether a respondent who listed brand attributes gave a "highly substantive" answer makes no conceptual sense. To enforce this constraint, I built a `ResponseTypeClassifier` that determines whether a response is opinion-based before the `SubstantivenessClassifier` runs. Both classifiers invoke a large language model (LLM) through the platform's shared `LlmAskerPicker` abstraction, which routes requests to the appropriate provider.

The question-level prerequisite was handled separately. A new `QuestionIntent` enum was added to the codebase with four values — `opinion`, `recall`, `entity_list`, and `attribute_list` — and a backfill script (`backfill_question_intent_opinion.py`) used LLM classification to retrospectively label all existing open-ended questions in MongoDB.

```python
# From bin/migrate/backfill_question_intent_opinion.py

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """You are an expert at analyzing survey
questions to determine their intent."""

INTENT_CLASSIFICATION_USER_PROMPT = """Your task is to determine the intent of the
question, i.e., what kind of responses to the survey question would be highly valuable
to the researcher. Select only one of the following intent categories that apply:

- entity_list: The researcher specifically wants the respondent to recall or list one
  or more entities, such as person, place, product, brand, or organization.
- attribute_list: The researcher specifically wants the respondent to share attributes
  or features they associate with a brand, product, person, or organization.
- recall: The researcher wants the respondent to recall specific information.
- opinion: The researcher wants the respondent to share their personal opinion,
  reaction, or evaluation.

Return your answer as JSON: {"intent": "<category>"}
"""
```

This prompt was crafted to be deterministic and exhaustive. Each category is described in terms of what the *researcher* wants, not what the *respondent* says — a subtle but important distinction that reduces ambiguous classifications on edge-case questions.

### Prompt Engineering with Jinja2 Templates

The substantiveness classifier uses Jinja2 templating to produce structured, context-aware prompts. The primary template (`substantiveness_opinion.jinja`) branches on `response_type` and applies a strict decision rubric:

```jinja2
{% if response_type == "opinion" %}
You are a reasoning model performing a *substantiveness classification* of a survey response.

## INPUTS
- Survey context: {{survey_context}}
- Survey question: {{survey_question}}
- Survey response: {{response}}

## DECISION RULES
- **Highly Substantive**: Contains a clear opinion AND at least one *thoughtful*
  supporting element: reasoning, explanation, evidence, examples, or suggestions.
- **Moderately Substantive**: Contains a clear opinion AND exactly one clear,
  simple reason. Lacks deeper elaboration.
- **Minimally Substantive**: Contains an opinion BUT the opinion is brief or
  surface-level AND no clear reason or explanation is provided.
- **Insubstantive**: Does not answer the question, shows no real opinion,
  is irrelevant, or is too vague to assess meaningfully.

**Tie-breaker rule:** If two categories seem plausible, choose the *less*
substantive of the two.

## OUTPUT FORMAT
Return a JSON dictionary ONLY:
{"category": "<one of: Highly Substantive, Moderately Substantive,
  Minimally Substantive, Insubstantive>"}
{% endif %}
```

Several design choices here were deliberate. First, the tie-breaker rule biases toward conservatism — an LLM that is uncertain should under-classify, because inflating substantiveness would give researchers a false sense of data quality. Second, the output format is restricted to a JSON dictionary with no surrounding reasoning, which makes parsing reliable and reduces token costs. Third, survey context and the original question text are injected into every prompt so the model can evaluate depth *relative to what was asked*, not in the abstract.

### Celery Task Orchestration and the Qualitative Analysis API

The classification logic runs as Celery tasks, following the platform's existing stateless task architecture. I added two new task constants — `DETERMINE_SUBSTANTIVENESS` and `DETERMINE_RESPONSE_TYPE` — and wired them into the task graph using Celery `chord` primitives, which allow a set of parallel classification tasks to fan out and then converge before downstream steps begin.

A new Flask endpoint, `POST /qualitative_analysis`, was added to expose the full classification pipeline as an async API:

```python
# From swayable_data/api/routes/qualitative_analysis.py

@API.route("/qualitative_analysis", methods=["POST"])
def qualitative_analysis() -> dict[str, str]:
    """
    Initiates asynchronous qualitative analysis for a survey.

    Runs all classification stages in order:
    - Backfill question intent
    - Relevance classification
    - Sentiment classification
    - Response type classification
    - Substantiveness classification (if opinion questions exist)
    """
    req_data = RequestData()
    survey_id = req_data.get_id("survey_id")

    runner = QualitativeAnalysisRunner(survey_id=survey_id)
    task_id = runner.run_async()

    status, result = TaskStatus.from_task_id(task_id)
    return {"status": status, "task_id": task_id}
```

The `QualitativeAnalysisRunner` encapsulates the full sequencing logic. Relevance must precede sentiment, and both must precede substantiveness — these dependencies are encoded structurally in the runner rather than documented as conventions, which makes them impossible to accidentally violate.

### Multiprocessing Batch Runner

For bulk backfilling and testing outside the Celery queue, I wrote `unified_classification_runner.py`, a standalone CLI script that uses Python's `multiprocessing` module with a parallelism factor of 20. The script uses the `rich` library to render real-time progress bars with spinners, percentage completion, elapsed time, and ETA — making long batch jobs observable and debuggable without tail-scrolling logs.

The MongoDB schema was updated to store the new fields: `intent` on the questions collection, and `response_type` and `substantiveness` within each response's qualitative subdocument array. Corresponding JSON Schema files were updated for validation.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

This project was tracked end-to-end through Linear (ticket ENG-1637) and GitHub (PR #2796). The pull request spans 57 changed files, 5,908 additions, and 92 commits, reflecting iterative development and code review with a senior engineer. I followed the platform's established architectural conventions — stateless Celery tasks, DAO abstractions for MongoDB access, typed Python function signatures, and modular prompt templates — rather than introducing new patterns where existing ones were sufficient. Helper scripts were written with `--dry-run` flags, `--limit` parameters for cost-controlled testing, and `--export-ids` options for saving intermediate state to CSV, all of which reflect professional attention to operability. A comprehensive `README.md` was written under `bin/runbook/documentation/` covering workflows, backfill procedures, and best practices for future teammates.

### 2. Fundamental NLP Algorithms and Concepts

The core task is **text classification** using a large language model as the classifier. Rather than training a supervised model, the system uses **prompt-based zero-shot classification** — a technique where the classification rubric is embedded directly in the prompt and the LLM generalizes from its pretraining. This approach trades the need for labeled training data (expensive to collect for niche survey content) against the cost and latency of LLM API calls. The two-stage pipeline also reflects a fundamental NLP pattern: **prerequisite filtering**. Applying a coarse classifier first (question intent, then response type) before running the more expensive fine-grained classifier (substantiveness) mirrors cascaded classification architectures common in production NLP systems. The system also handles multilingual input — the substantiveness prompt explicitly instructs the model to detect and translate non-English responses before classifying, addressing a real distributional challenge in survey data.

### 3. Tools and Packages

- **Large Language Models (LLM)**: The platform's `LlmAskerPicker` abstraction routes classification requests to an underlying LLM provider. Prompts were engineered specifically for instruction-tuned models expected to return structured JSON.
- **Jinja2**: Used for templating classification prompts, enabling dynamic injection of survey context, question text, and response content, and branching on `response_type` within a single template file.
- **Celery**: The distributed task queue used for async classification jobs, with `chord` primitives for fan-out/fan-in task graphs.
- **MongoDB / PyMongo**: The document database backing the platform; I performed schema migrations adding new fields to both the questions and responses collections, and queried MongoDB for batches of unclassified responses.
- **Rich**: Python library used in the batch runner to render real-time terminal progress bars with spinner animations and ETA estimates.
- **Click**: Used for building the CLI interface of the unified classification runner and helper scripts.
- **Flask**: Web framework for the new `POST /qualitative_analysis` REST endpoint.
- **uv**: The project's Python package and environment manager, used to run scripts with environment variable injection.

## Challenges and Solutions

The most technically demanding challenge was determining the correct sequencing and gating logic for a multi-stage classification pipeline where each stage depends on upstream results. Early in development, it was unclear whether substantiveness classification should run unconditionally or only when opinion-type questions were detected. Running it unconditionally wastes LLM tokens on recall and entity-listing questions, for which the concept of substantiveness is arguably undefined. Running it conditionally requires the system to know, at dispatch time, whether any opinion questions exist in a given survey.

The solution was to encode the intent prerequisite at two levels. At the question level, the `backfill_question_intent_opinion.py` script populates the `intent` field in MongoDB before any response-level classification begins, so downstream tasks can query for opinion questions without any additional LLM calls. At the response level, the `ResponseTypeClassifier` acts as a per-response gate: even if a question is labeled `opinion`, individual responses might be observations, non-answers, or test submissions, and these should not receive a substantiveness score.

A secondary challenge was cost control during development. The LLM API charges per token, and a full survey can have thousands of responses. To mitigate this, every helper script was built with a `--limit` flag that stops after classifying N responses, and a `--dry-run` mode that queries MongoDB without making any LLM calls or writes. These flags were essential for rapid iteration during prompt development.

## Outcomes and Impact

The pull request was merged into `main` on February 9, 2026, completing the ENG-1637 ticket. The feature delivers a new dimension of qualitative signal — response substantiveness — that is now available for every opinion question in the Swayable platform. Researchers can filter survey results to surface only `highlySubstantive` responses, reducing the manual effort required to identify analytically useful answers in large open-end datasets. The new `POST /qualitative_analysis` endpoint integrates substantiveness into the existing automated classification pipeline alongside relevance and sentiment, so future surveys receive the full suite of qualitative labels without any additional operator intervention. The addition of the `QuestionIntent` schema field also lays groundwork for future classifiers that may behave differently depending on what a question is designed to measure.

## Code Reference

The work for this project is in a private repository. The pull request is available to authorized viewers here:

[View Pull Request on GitHub — ENG-1637 Substantiveness Classification](https://github.com/swayable/swayable-data/pull/2796)

Meaningful code snippets from the LLM prompt template, the backfill script, and the API endpoint are embedded in the Technical Approach section above.
