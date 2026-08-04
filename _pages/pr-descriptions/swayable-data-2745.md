---
title: "swayable-data #2745 — Cron-based sentiment and relevance"
permalink: /pr-descriptions/swayable-data-2745/
author_profile: false
---

Sanitized summary of a private pull request for portfolio review.

**Original PR (private / org-restricted):** [https://github.com/swayable/swayable-data/pull/2745](https://github.com/swayable/swayable-data/pull/2745)

**Repository:** `swayable/swayable-data` · **PR number:** 2745  
**Original title:** feat: cron based sentiment and relevance  
**Ticket:** ENG-1268

---

### Overview

Continues prior work to run qualitative sentiment and relevance classification on a schedule.

Cron scheduling is managed by Celery (task configuration module in the data service).

### Finalization Policy for Sentiment and Relevance

- Stateless cron runs on a fixed interval
- Database queried for unprocessed responses
- Each response is processed via Celery tasks `run_relevance_classification` and `run_sentiment_classification`
- Failures are logged and picked up automatically on the next cron run

### Queue Processing / Retry Policy

- Responses are added to the queue when the cron kicks off and queries unprocessed responses for active surveys
- The cron picks up responses on each interval
- If a task fails, the error is logged, the worker continues gracefully, and the response is left for retry on the next cycle
- Already-processed responses are skipped via exclude-preprocess task checks in the classify helpers

### Changes

Scheduled Celery-driven sentiment and relevance classification with durable retry semantics for unprocessed open-ended responses on active surveys.
