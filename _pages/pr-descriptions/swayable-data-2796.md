---
title: "swayable-data #2796 — Substantiveness classification pipeline"
permalink: /pr-descriptions/swayable-data-2796/
author_profile: false
---

Sanitized summary of a private pull request for portfolio review.

**Original PR (private / org-restricted):** [https://github.com/swayable/swayable-data/pull/2796](https://github.com/swayable/swayable-data/pull/2796)

**Repository:** `swayable/swayable-data` · **PR number:** 2796  
**Original title:** ENG-1637 substantiveness classification  
**Ticket:** ENG-1637

---

### Overview

Implement substantiveness classification for opinion questions (`question.intent="opinion"`). Accomplished in the classification pipeline as well as a unified classification runner script under `bin/runbook/`. Helper scripts support local testing (unset classifications, backfill question intent, accuracy checks).

**Important:** Substantiveness classification depends on `question.intent="opinion"`. The schema migration must be run as well as backfilling question intent (optionally for a single survey).

### Core Classification

- New `SubstantivenessClassifier` classifies responses as: `highlySubstantive`, `moderatelySubstantive`, `minimallySubstantive`, or `insubstantive`
- New `ResponseTypeClassifier` determines if responses are opinion-based (prerequisite for substantiveness)
- Added `DETERMINE_SUBSTANTIVENESS` and `DETERMINE_RESPONSE_TYPE` task constants

### Question Intent System

- Added `QuestionIntent` enum: `opinion`, `recall`, `entity_list`, `attribute_list`

### API and Orchestration

- New `/qualitative_analysis` POST endpoint for async qualitative analysis
- New `QualitativeAnalysisRunner` orchestrates all classification stages
- New `unified_classification_runner.py` script runs all classifications with multiprocessing (parallelism=20) and real-time progress tracking

### Schema Updates

- Added `intent` field to questions collection
- Added `response_type` and `substantiveness` fields to responses qualitative array
- Added task flags: `response_type_classification`, `substantiveness_classification`

### Helper Scripts

- `backfill_question_intent_opinion.py` — LLM-based question intent classification with `--limit` for smaller testing data sets
- `unset_openend_classification.py` — Remove classifications
- `export_opinion_questions_for_testing.py` — Export questions to YAML for manual labeling
- `test_substantiveness_accuracy.py` — Test classifier accuracy against labeled data
- `test_openend_classification_filter.py` — Diagnostic tool for queue filtering

### LLM Integration

- Added Jinja2 prompt templates: `substantiveness.jinja`, `substantiveness_opinion.jinja`, `substantiveness_entity.jinja`
- Implemented `PrompterSubstantiveness` and `ParserSubstantivenessBasic`

### Documentation

- Runbook-style README under `bin/runbook/documentation/` covering workflows and best practices

### Testing (sanitized)

- Confirm API-service schema migration
- Test with a demo survey locally
- Verify MongoDB documents have `intent` where expected
- Backfill question intent for a survey, optionally unset classifications, then run the unified classification runner
