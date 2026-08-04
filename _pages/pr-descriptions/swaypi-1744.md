---
title: "swaypi #1744 — Sampling field on survey models"
permalink: /pr-descriptions/swaypi-1744/
author_profile: false
---

Sanitized summary of a private pull request for portfolio review.

**Original PR (private / org-restricted):** [https://github.com/swayable/swaypi/pull/1744](https://github.com/swayable/swaypi/pull/1744)

**Repository:** `swayable/swaypi` · **PR number:** 1744  
**Original title:** feat(survey): add sampleDescription field to survey and surveyDraft models  
**Ticket:** COR-275  
**Related:** [ui #2307](https://github.com/swayable/ui/pull/2307) ([local sanitized page](/pr-descriptions/ui-2307/))

---

### Overview

Adds `sampleDescription` as an optional string field (max 1000 chars) to both the `SurveyDraft` and `Survey` Mongoose models, transfers the value from draft to survey on launch, registers permissions, and updates the local JSON schema files used by the schema upgrade script.

**Schema-validator note (sanitized):** This PR also adds auto-sync of MongoDB JSON-schema validators at API boot in non-production environments. CI's preloaded MongoDB image is baked from main, so its validators reject writes of any field added on a branch (`additionalProperties: false`) — Mongoose reads succeed but UI-driven end-to-end writes fail with document validation errors. Schema checks now auto-synchronize validators in non-prod so test/dev validators stay aligned with each branch's schema JSON (no-op when already current; production/staging unchanged). This unblocks future PRs that add a user-facing field with an end-to-end test exercising the write path.

### Changes

- Survey draft and survey Mongoose models — add `sampleDescription`
- Create-survey-from-draft command — copy `sampleDescription` from draft to survey on launch
- Local JSON schemas for drafts and surveys — add the field for schema upgrade
- Boot-time schema check — auto-sync validators in non-production
- GraphQL authorization permissions for draft and survey — `sampleDescription: writers`
- Updated permissions snapshot
- Two new unit tests: copies `sampleDescription` on launch; does not set it when the draft has none

### Testing (sanitized)

- Unit tests for create-survey-from-draft and permissions snapshots pass
- GraphQL types are auto-generated from the Mongoose schema (no manual SDL changes)
