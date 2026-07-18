---
title: "Sampling Information, End to End (~15–20 hours)"
excerpt: "<strong>Estimated effort: ~15–20 hours.</strong> Added sampleDescription across database schemas, GraphQL, launch and setup interfaces, and Cypress coverage while diagnosing stale test database validators."
collection: portfolio
---

**Estimated effort: ~15–20 hours** (COR-275; UI PR #2307 and swaypi PR #1744)

## Project Overview

Swayable clients configure surveys before launch, but they previously had no structured place to communicate custom sampling needs. Customer Success administrators also needed to review or revise those instructions after launch without exposing internal sampling information on the client-facing status page. I implemented `sampleDescription` as an optional field that follows the survey from draft configuration into the launched survey and remains editable in the administrative setup interface.

This project was an end-to-end feature rather than an isolated form control. The value had to exist on both `SurveyDraft` and `Survey`, survive draft-to-survey conversion, pass GraphQL authorization, appear in two different editing contexts, remain absent from the client status query, and be exercised through the browser against a real database validator. The merged UI pull request contained 589 additions across 13 files, and the API pull request contained 200 additions across 10 files.

The launch interface presents “Standard Sampling” and “Custom Sample Instructions” options. Selecting the custom option reveals a bounded text area and character counter. After launch, an administrator can edit the same value in a dedicated Sampling section on the setup page. The status page intentionally does not select the field. This distinction encoded the product requirement in query construction rather than depending only on hiding a rendered component.

## Technical Approach

I began at the persistence layer. I added an optional string field with a 1,000-character maximum to the Mongoose definitions for drafts and surveys. I kept the MongoDB JSON validators synchronized with those model changes and registered the field with the standard writer permissions for both entity types. The launch command then copied the value from the draft to the survey only when it was present.

A sanitized representation of the transfer behavior is:

```javascript
const surveyInput = {
  title: draft.title,
  organizationId: draft.organizationId,
}

if (draft.sampleDescription) {
  surveyInput.sampleDescription = draft.sampleDescription
}
```

The production command uses the repository data structures and identifiers, but the important behavior is that the optional field is preserved without manufacturing a value for drafts that do not contain one. Unit tests covered both cases: a custom description is copied at launch, and an absent description remains absent.

In the UI, I added the field to the launch-page GraphQL selection and built a shared sampling-type toggle. The launch form and administrative setup section reuse that presentational control while connecting it to different state-management paths. On the setup page, edits flow through the existing pending-changes mechanism rather than bypassing established save behavior.

I also applied least-exposure query design. The administrative query builder selects `sampleDescription`, while the client query builder does not. This makes the boundary visible in code:

```javascript
const adminFields = ["title", "status", "sampleDescription"]
const clientFields = ["title", "status"]
```

This snippet is simplified and contains no private schema details, but it captures the implemented separation. A client-facing status page cannot accidentally render data that its query never retrieves.

Automated coverage included component tests for toggle state, text-area visibility, disabled behavior, and pending-change updates. Cypress coverage exercised switching to custom instructions, entering a value, saving, and confirming persistence. The browser test was particularly valuable because it crossed every layer from Vue through GraphQL and the API into MongoDB.

That end-to-end test exposed an infrastructure issue. CI used a preloaded MongoDB image built from the main branch. Its JSON-schema validators enforced `additionalProperties: false`, so the branch model could read the new field but writes were rejected because the running validator did not yet include it. I traced the `DocumentValidationFailure` to the absence of schema synchronization in the end-to-end test startup path. I documented a proposed recipe-level fix in closed sway PR #49. The merged API solution instead made `checkSchemas()` synchronize validators automatically in non-production environments. I claim the diagnosis and the merged API change in swaypi #1744. I do not claim later changes to sway repository recipes made by other contributors.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

I implemented a production feature across two repositories, coordinated matching branches, responded to integration failures, and wrote review-ready descriptions with explicit QA steps. The work followed existing component, store, GraphQL, authorization, schema, and command patterns. The result demonstrates the workplace skill of completing the entire data path rather than stopping when the form renders.

### 2. Data Modeling and Application Integration

I learned how one conceptual field can require several concrete representations. `sampleDescription` existed in Mongoose models, MongoDB validators, generated GraphQL types, authorization rules, query fragments, draft conversion, UI state, and tests. Each representation had a different failure mode. Maintaining agreement among them was the central systems-learning outcome.

### 3. Human Language Technology and NLP Concepts

This project handled human-authored natural-language instructions, but it did not perform NLP. There was no parsing, classification, embedding, generation, or linguistic evaluation of the sample description. The relevant HLT lesson was responsible text-data handling: set a clear length constraint, preserve author intent through transformations, and restrict the text to the operational audience that needs it. I do not present this as an NLP algorithm project.

### 4. Testing and Diagnostic Reasoning

I combined unit, component, and Cypress tests. Unit tests checked draft conversion, component tests checked interaction states, and Cypress verified persistence through the deployed stack. When the Cypress write failed, I compared the branch-local schema files with the validator loaded by CI and identified why Mongoose reads could succeed while MongoDB writes failed. This distinction between application model and database enforcement was a significant diagnostic lesson.

## Challenges

The main challenge was the stale validator in CI. Evidence from the failing test showed `DocumentValidationFailure` and identified `sampleDescription` as an additional property. The API branch already contained the updated Mongoose model and JSON schema files, so the failure could not be explained by an omitted field definition. Investigation showed that the preloaded test database retained validators from main and that the test startup path did not apply branch schemas.

The closed sway PR #49 documents the diagnosis and one proposed fix: running schema upgrade before Cypress. The final merged solution was in swaypi #1744, where non-production schema checks call the existing synchronizer and remain a no-op when validators are current. Production and staging behavior were not changed. This solution addressed the underlying branch-schema mismatch for future user-facing fields as well as this feature.

A second challenge was preserving audience boundaries. The requirement said that sampling information should be editable by administrators but absent from the client status page. I implemented this at the GraphQL fragment level and documented it in the PR quality checklist. The evidence supports the narrower claim that the field is omitted from the client-facing query, not a claim of a new platform-wide security subsystem.

## Outcomes

Both principal pull requests merged on May 21, 2026. Clients can provide custom sampling instructions while configuring a draft. The value transfers to the survey at launch, administrators can edit it in setup, and the client status page does not request it. The merged tests protect the optional transfer and primary UI interaction paths.

The schema synchronization change also improved local and CI reliability for branches that add database fields exercised by browser tests. It reduced a class of failures in which application schemas and test database validators diverge. The scope of my claim is the diagnosis and the non-production synchronization merged in swaypi #1744.

## Professional Practice

I treated acceptance criteria as data-flow and audience constraints, not only visual requirements. I documented dependencies between repositories, supplied commands for targeted test suites, and described why the Cypress failure occurred. I also revised the implementation path when the recipe-level proposal was closed, placing the merged safeguard in the API schema-checking lifecycle.

The project reinforced careful attribution. The related sway #49 record is useful evidence for the diagnosis, but it was closed. Later recipe work in that repository is outside my contribution. Separating investigation, merged code, and subsequent work by others is essential in a professional portfolio.

## Code Reference

The repositories are private and available to authorized viewers:

- [UI #2307 — launch and setup sampling interfaces](https://github.com/swayable/ui/pull/2307)
- [swaypi #1744 — models, transfer, permissions, tests, and non-production schema synchronization](https://github.com/swayable/swaypi/pull/1744)
- [sway #49 — closed proposal documenting the schema auto-sync diagnosis](https://github.com/swayable/sway/pull/49)

Representative files include `components/SamplingTypeToggle.vue`, `components/surveySetup/SurveySetupSampling.vue`, the launch page, administrative GraphQL fragments, Cypress setup sampling coverage, both Mongoose schemas, draft conversion tests, authorization definitions, and the API database initialization module.
