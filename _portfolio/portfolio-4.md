---
title: "Substantiveness Classification for Open-End Survey Responses"
excerpt: "Built a multi-stage LLM classification pipeline that evaluates the depth and quality of open-ended survey responses, categorizing them into four tiers of substantiveness. The system spans two microservices, introduces a new MongoDB schema dimension, and integrates with an existing async classification infrastructure via Celery task orchestration.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

## Project Overview

Swayable is a research platform that measures persuasion through online surveys. Surveys frequently include open-ended questions that invite respondents to share opinions in their own words. Historically, the platform classified these responses for relevance and sentiment — but lacked any measure of *depth*. A response that says "I liked it" and one that says "I found the pacing compelling because it reminded me of how long-form journalism creates emotional investment" were treated identically, despite their vastly different analytical value.

This ticket, ENG-1637, introduced **Substantiveness Classification**: an automated system that evaluates the intellectual depth of survey responses and assigns each one to one of four categories — `highlySubstantive`, `moderatelySubstantive`, `minimallySubstantive`, or `insubstantive`. The classifier targets opinion-type questions specifically, distinguishing them from recall, entity-listing, or attribute-listing question types. The expected outcome was a new dimension of qualitative data that allows researchers to filter for high-signal responses, surface the most analytically useful answers at scale, and give clients a richer picture of audience engagement with their content.

The project required end-to-end implementation across two microservices: schema changes in the Node.js/MongoDB `swaypi` service and a full NLP pipeline in the Python `swayable-data` service, including prompt engineering, a new Celery task layer, a REST API endpoint, and a command-line batch runner for backfilling and testing. It was completed and merged on February 9, 2026.

---

## System Architecture

The feature spans two backend services and a Celery task queue. Data flows from an API request all the way to per-response MongoDB documents:

```
[UI / API Client]
      │  POST /qualitative_analysis { survey_id }
      ▼
[swayable-data: Flask API]          qualitative_analysis.py
      │  QualitativeAnalysisRunner.run_async()
      ▼
[Celery Task Queue]                 run_qualitative_analysis_task
      │
      ├── Stage 0: backfill_question_intent        (LLM → intent="opinion")
      │             ↓ writes to: questions.intent
      │
      ├── Stage 1: RelevanceClassifier
      ├── Stage 2: SentimentClassifier
      ├── Stage 3a: ResponseTypeClassifier          (LLM → response_type)
      │             ↓ writes to: responses[].qualitative[].response_type
      │
      └── Stage 3b: SubstantivenessClassifier       (only if opinion_count > 0)
                    │  substantiveness_opinion.jinja
                    ▼
             [LlmAskerPicker → LLM Provider]
                    ↓ writes to: responses[].qualitative[].substantiveness

[swaypi: Node.js/MongoDB]
      Schema migration: questions.intent (enum field)
                        responses[].qualitative[].substantiveness
                        responses[].qualitative[].response_type
```

Stage 3b runs only when Stage 0 confirms that the survey contains at least one question with `intent = "opinion"`. This gating prevents unnecessary LLM calls for surveys composed entirely of recall, entity-listing, or attribute-listing questions.

---

## Technical Approach

### Two-Stage Classification Architecture

The most important design decision in this project was introducing a prerequisite classification step before substantiveness scoring. Substantiveness is only meaningful for opinion-type responses — asking whether a respondent who listed brand attributes gave a "highly substantive" answer makes no conceptual sense. To enforce this constraint, I built a `ResponseTypeClassifier` that determines whether a response is opinion-based before the `SubstantivenessClassifier` runs. Both classifiers invoke a large language model through the platform's shared `LlmAskerPicker` abstraction, which routes requests to the appropriate provider.

The question-level prerequisite was handled separately. A new `QuestionIntent` enum was added to the codebase with four values — `opinion`, `recall`, `entity_list`, and `attribute_list` — and a backfill script used LLM classification to retrospectively label all existing open-ended questions in MongoDB. The `classify_question_intent` function below handles a single question:

```python
# From bin/migrate/backfill_question_intent_opinion.py

INTENT_CLASSIFICATION_USER_PROMPT = """Your task is to determine the intent of the
question, i.e., what kind of responses to the survey question would be highly valuable
to the researcher. Select only one of the following intent categories that apply:

- entity_list: The researcher specifically wants the respondent to recall or list one
  or more entities...
- attribute_list: The researcher specifically wants the respondent to share attributes
  or features they associate with a brand, product, person or organization.
- recall: The researcher wants the respondent to recall specific information, details,
  elements, or aspects ABOUT THE STIMULUS ITSELF...
- opinion: The researcher wants the respondent's personal opinion, belief, viewpoint,
  reaction, feedback, or thoughts about entities, concepts, or topics...

Return your final answer in the JSON format {{"Thought process": "...", "Category": "category"}}.

Survey context: {survey_context}
Survey question: {question_text}"""


def classify_question_intent(question_text: str, survey_context: str, asker: Any) -> str:
    """Use LLM to determine intent of a question."""
    prompt = INTENT_CLASSIFICATION_USER_PROMPT.format(
        question_text=question_text,
        survey_context=survey_context,
    )
    result = asker.ask(prompt, INTENT_CLASSIFICATION_SYSTEM_PROMPT)

    if not result.request_was_successful():
        print(f"Warning: LLM classification failed for: {question_text[:50]!r}")
        return ""

    response_text = result.get_response_text().strip()

    # Extract JSON if inside code blocks
    if "```" in response_text:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end]

    try:
        data = json.loads(response_text)
        category: str = str(data.get("Category", "")).lower()
        valid_intents = [intent.value for intent in QuestionIntent]
        if category in valid_intents:
            return category
        return ""
    except json.JSONDecodeError:
        return ""
```

This prompt was crafted deliberately. Each category is described in terms of what the *researcher* wants, not what the *respondent* says — a subtle but important distinction that reduces ambiguous classifications on edge-case questions. The categories are mutually exclusive and exhaustive by design.

### Prompt Engineering with Jinja2 Templates

The substantiveness classifier uses Jinja2 templating to produce structured, context-aware prompts. The primary template (`substantiveness_opinion.jinja`) branches on `response_type` and applies a strict decision rubric for opinion responses:

```jinja2
{# From swayable_data/io/llm/templates/substantiveness_opinion.jinja #}

{% if response_type == "opinion" %}
You are a reasoning model performing a *substantiveness classification* of a survey response.

## INPUTS
- Survey context: {{survey_context}}
- Survey question: {{survey_question}}
- Survey response: {{response}}

## DECISION RULES
- **Highly Substantive**
  - Contains a clear opinion AND at least one thoughtful supporting element:
    reasoning, explanation, evidence, examples, or suggestions.

- **Moderately Substantive**
  - Contains a clear opinion AND exactly one clear, simple reason.
  - Lacks deeper elaboration.

- **Minimally Substantive**
  - Contains an opinion BUT the opinion is brief or surface-level,
    AND no clear reason or explanation is provided.

- **Insubstantive**
  - Does not answer the question, shows no real opinion, is irrelevant,
    or is too vague to assess meaningfully.

**Tie-breaker rule:**
If two categories seem plausible, choose the *less* substantive of the two.

## OUTPUT FORMAT
Return a JSON dictionary ONLY:
{"category": "<one of: Highly Substantive, Moderately Substantive,
  Minimally Substantive, Insubstantive>"}
{% endif %}
```

Several design choices here were deliberate. First, the tie-breaker rule biases toward conservatism — an LLM that is uncertain should under-classify, because inflating substantiveness would give researchers a false sense of data quality. Second, the output format is restricted to a JSON dictionary with no surrounding reasoning, which makes parsing reliable and reduces token costs. Third, survey context and the original question text are injected into every prompt so the model can evaluate depth *relative to what was asked*, not in the abstract.

### Celery Task Orchestration

The classification logic runs as a Celery task, following the platform's existing stateless task architecture. The `run_qualitative_analysis_task` function encodes all stage dependencies directly in code — relevance must precede sentiment, both must precede substantiveness — rather than documenting them as conventions:

```python
# From swayable_data/services/classification/qualitative_analysis_runner.py

@APP.task(base=BSONSingleton, bind=True)
def run_qualitative_analysis_task(self, survey_id: ObjectId):
    """Celery task to run all qualitative analysis classifications for a survey."""

    # Stage 0: Backfill question intent (prerequisite for substantiveness)
    opinion_question_count = _backfill_question_intent(survey_id)

    # Stage 1: Relevance Classification
    relevance_classifier = RelevanceClassifier(survey_id, limit=10000)
    relevance_classifier.classify_and_save_relevance()

    # Stage 2: Sentiment Classification
    sentiment_classifier = SentimentClassifier(survey_id, limit=10000)
    sentiment_classifier.classify_and_save_sentiment()

    # Stage 3a: Response Type Classification (prerequisite for substantiveness)
    response_type_runner = ResponseTypeClassificationRunner(survey_id, limit=10000)
    response_type_runner.run()

    # Stage 3b: Substantiveness Classification (conditional on opinion questions)
    if opinion_question_count > 0:
        substantiveness_runner = SubstantivenessClassificationRunner(survey_id, limit=10000)
        substantiveness_runner.run()

    return {
        "status": "SUCCESS",
        "opinion_question_count": opinion_question_count,
        "ran_substantiveness": opinion_question_count > 0,
    }
```

The `SubstantivenessClassifier` itself follows the platform's strategy pattern, using a `ClassificationConfig` to gate by both question intent and response type:

```python
# From swayable_data/services/classification/classifiers/quals/substantiveness_classifier.py

class SubstantivenessClassifier:
    """Classify the substantiveness of responses to long-form opinion questions."""

    def __init__(self, survey_id, analysis=None, limit=MAXIMUM_COMMENTS,
                 response_ids=None, open_ends_set=None):
        self.survey_id = survey_id
        self.analysis = analysis or dao.get_analysis(survey_id)

        # Build config gating by BOTH question intent and response type
        config = ClassificationConfig(
            process_with_llm=llm_config.process_with_llm(),
            max_comments=limit,
            filter_by_intent=QuestionIntent.OPINION,      # question-level gate
            filter_by_response_type="opinion",            # response-level gate
        )

        self.strategy = SubstantivenessStrategy(
            survey_id=survey_id,
            config=config,
            analysis=self.analysis,
            open_ends_set=open_ends_set,
        )

    def classify_all(self):
        """Determine the substantiveness of each response."""
        return self.strategy.execute()
```

The dual gating — `filter_by_intent` at the question level, `filter_by_response_type` at the response level — ensures the LLM is only invoked for responses that are genuinely opinion-based, preventing wasted API calls and incorrect classifications on non-opinion content.

### API Endpoint and Batch Runner

A new Flask endpoint, `POST /qualitative_analysis`, exposes the full pipeline as an async API, returning a Celery task ID immediately so the caller does not block. A separate `unified_classification_runner.py` CLI script uses Python's `multiprocessing` module with a parallelism factor of 20 for bulk backfilling outside the Celery queue, using the `rich` library to render real-time progress bars with ETA estimates.

---

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

This project was tracked end-to-end through Linear (ticket ENG-1637) and GitHub (PR #2796). The pull request spans 57 changed files, 5,908 additions, and 92 commits, reflecting iterative development with code review from a senior engineer. I followed the platform's established architectural conventions — stateless Celery tasks, DAO abstractions for MongoDB access, typed Python function signatures, the strategy pattern for classifiers, and modular Jinja2 prompt templates — rather than introducing new patterns where existing ones were sufficient. Every helper script was built with `--dry-run` flags and `--limit` parameters for cost-controlled testing. A comprehensive `README.md` was written under `bin/runbook/documentation/` covering workflows, backfill procedures, and best practices for future teammates.

### 2. Fundamental NLP Algorithms and Concepts

The core task is **text classification** using a large language model as the classifier. Rather than training a supervised model, the system uses **zero-shot prompt-based classification** — the classification rubric is embedded directly in the prompt and the LLM generalizes from its pretraining. This approach trades the need for labeled training data (expensive to collect for niche survey content) against the cost and latency of LLM API calls. The two-stage pipeline also reflects a fundamental NLP pattern: **cascaded classification**. Applying a coarse classifier first (question intent, then response type) before running the fine-grained classifier (substantiveness) mirrors production NLP systems where cheaper upstream filters reduce the load on more expensive downstream models. The system also handles multilingual input — the prompt explicitly instructs the model to detect and translate non-English responses before classifying.

### 3. Tools and Packages

- **Jinja2** — Templating for dynamic, context-aware LLM prompts with branching on `response_type`
- **Celery** — Distributed async task queue; the full classification pipeline runs as a single `@APP.task`
- **MongoDB / PyMongo** — Document store; schema migrations added `intent` to questions and `response_type`/`substantiveness` to response qualitative subdocuments
- **Rich** — Real-time terminal progress bars with spinner and ETA in the batch runner CLI
- **Click** — CLI interface for helper and backfill scripts
- **Flask** — Web framework for the new `POST /qualitative_analysis` endpoint
- **uv** — Project Python package manager used to run scripts with environment injection

---

## Challenges and Solutions

The most technically demanding challenge was determining the correct sequencing and gating logic for a multi-stage pipeline where each stage depends on upstream results. Early in development, it was unclear whether substantiveness classification should run unconditionally or only when opinion-type questions were detected. Running it unconditionally wastes LLM tokens on recall and entity-listing questions, for which the concept of substantiveness is arguably undefined. Running it conditionally requires the system to know, at dispatch time, whether any opinion questions exist in a given survey.

The solution was to encode the intent prerequisite at two levels. At the question level, Stage 0 of the Celery task populates the `intent` field in MongoDB before any response-level classification begins, so downstream stages can query for opinion questions without additional LLM calls. At the response level, the `ResponseTypeClassifier` acts as a per-response gate: even if a question is labeled `opinion`, individual responses might be non-answers, test submissions, or observations, and these should not receive a substantiveness score.

A secondary challenge was cost control during development. The LLM API charges per token, and a full survey can have thousands of responses. Every helper script was built with a `--limit` flag that stops after classifying N responses and a `--dry-run` mode that queries MongoDB without making any LLM calls or writes. These flags were essential for rapid iteration during prompt development.

---

## Outcomes and Impact

The pull request was merged into `main` on February 9, 2026. The feature delivers a new dimension of qualitative signal — response substantiveness — that is now available for every opinion question in the Swayable platform. Researchers can filter survey results to surface only `highlySubstantive` responses, reducing the manual effort required to identify analytically useful answers in large open-end datasets. The new `POST /qualitative_analysis` endpoint integrates substantiveness into the existing automated classification pipeline alongside relevance and sentiment, so future surveys receive the full suite of qualitative labels without additional operator intervention. The `QuestionIntent` schema field also lays groundwork for future classifiers that may behave differently depending on what a question is designed to measure.

---

## Code Reference

The work for this project is in a private repository. The pull request is available to authorized viewers here:

[View Pull Request on GitHub — ENG-1637 Substantiveness Classification](https://github.com/swayable/swayable-data/pull/2796)

Because the repository is private, the key implementation files are reproduced inline above: the intent classification backfill script (`bin/migrate/backfill_question_intent_opinion.py`), the Jinja2 prompt template (`swayable_data/io/llm/templates/substantiveness_opinion.jinja`), the Celery task orchestrator (`swayable_data/services/classification/qualitative_analysis_runner.py`), and the classifier entry point (`swayable_data/services/classification/classifiers/quals/substantiveness_classifier.py`).
