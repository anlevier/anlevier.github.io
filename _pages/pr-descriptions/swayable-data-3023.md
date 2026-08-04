---
title: "swayable-data #3023 — Results email on final analysis"
permalink: /pr-descriptions/swayable-data-3023/
author_profile: false
---

Sanitized summary of a private pull request for portfolio review.

**Original PR (private / org-restricted):** [https://github.com/swayable/swayable-data/pull/3023](https://github.com/swayable/swayable-data/pull/3023)

**Repository:** `swayable/swayable-data` · **PR number:** 3023  
**Original title:** fix: results email send on finalization  
**Ticket:** COR-46

---

### Overview

The "Your results are ready" email was sent when an admin clicked the finalize button, rather than when final analysis completed. This change ties notification enqueue to final analysis completion (`results_marked_final = True`) in the analysis task path.

### Changes

- Analysis task module — send / queue results notification when analysis is marked final, instead of at the earlier finalize-button moment

### Testing (sanitized)

Manual local verification on a demo survey (identifiers and customer labels omitted):

1. Reset notification and analysis-meta state for the survey in local MongoDB
2. Seed the survey as finalized with a pending analysis run and a successful logged analysis task
3. Invoke `update_analysis_meta` for the survey and confirm it reaches a finalized-results outcome
4. Invoke notification queueing for the survey and confirm the results-ready notification is created as expected

Internal chat threads, absolute machine paths, connection strings, ObjectIds, and customer-named demo labels from the original PR checklist were removed for this public summary.
