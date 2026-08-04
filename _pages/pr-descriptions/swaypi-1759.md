---
title: "swaypi #1759 — Quiz item generation for text-span annotation"
permalink: /pr-descriptions/swaypi-1759/
author_profile: false
---

Sanitized summary of a private pull request for portfolio review.

**Original PR (private / org-restricted):** [https://github.com/swayable/swaypi/pull/1759](https://github.com/swayable/swaypi/pull/1759)

**Repository:** `swayable/swaypi` · **PR number:** 1759  
**Original title:** feat: append articleHighlighter quiz item in getQuizBySurveyId  
**Sanitized title focus:** Append a text-span annotation quiz page when generating a quiz from a survey  
**Ticket:** COR-424 (API half of respondent text-span annotation; unblocks COR-427 UI wrapper)

---

### Overview

Implements the API half of the respondent text-span annotation experience. Unblocks the UI quiz wrapper integration. Independent of a related schema PR — the two can land in either order. No deploy/migration step.

### Changes

- Add a quiz item type constant for text-span annotation
- When converting a survey to a quiz, append an annotation page when the content config enables the feature **and** the assigned treatment is a **non-placebo** markdown article
  - **Reused page builder, not a new one:** the page is produced by the existing content-to-page path with a new content option that only swaps the emitted item type (generic content → annotation). No parallel rendering code
  - **Why the treatment content, not a separate selection:** the text being annotated *is* the treatment article, so the page references the survey treatment directly. Persisting which content was annotated into the interview record is a separate concern handled at response time
  - **Placebo excluded:** control respondents do not get the annotation page, mirroring engagement collection. Highlights are character offsets into the treatment article's text and carry no analytical value for the control arm
- Preview quiz generation inherits the behavior through the shared survey-to-quiz path; no preview-side code change

### Testing (sanitized)

- New unit coverage for: toggle on + markdown treatment (appends page); toggle on + video treatment (no page); toggle off + markdown (no page); toggle on + placebo markdown (no page); engagement + annotation both on (both pages present)
- Preview path asserts the annotation item when the draft uses markdown content
- Extended preview-shape whitelist to include the new item type
- Manual confirmation on a draft with the feature enabled and markdown content; confirmation that a non-markdown treatment does not produce the page
