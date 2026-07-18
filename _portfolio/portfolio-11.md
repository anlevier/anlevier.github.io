---
title: "Dial Testing Activation (~10–14 hours)"
excerpt: "<strong>Estimated effort: ~10–14 hours.</strong> Added the schema and eligibility-gated survey-editor control that activates dial-test collection only for surveys containing an eligible video treatment."
collection: portfolio
---

**Estimated effort: ~10–14 hours** (COR-539 and COR-542; swaypi PR #1772 and UI PR #2388)

## Project Overview

Dial Testing is a broader Swayable capability that records a respondent signal while video content plays. My contribution was the enablement slice: I added the persisted survey setting and the author-facing control that allows eligible surveys to opt into collection. I did not build the respondent dial, time-series capture, analysis pipeline, or the complete Dial Testing product.

Before this work, downstream quiz-building services had no survey field that represented whether Dial Testing should be enabled. Survey authors also had no setup control for changing that state. The two tickets established a narrow contract: `contentConfig.enableDialTest` is optional, an absent or false value means off, and the editor displays “Collect Dial Test Data” only when the survey includes at least one non-placebo video treatment. The control is additionally protected by the `dial-test` feature flag.

The eligibility rule matters because a dial interaction is meaningful only when the survey contains the applicable stimulus. Placebo content is excluded, and non-video media does not qualify. This prevented authors from activating an unsupported configuration while keeping the stored setting available to the downstream services that implement the respondent experience.

## Technical Approach

I implemented the backend contract first because the UI GraphQL fragments could not query a field that the API did not expose. In the survey content configuration schema, I added an optional Boolean that mirrors established fields such as engagement and article-highlighter enablement. I then added the same property to the MongoDB JSON validator and registered it with the standard writer authorization template. The GraphQL type is generated from the Mongoose model, so keeping the model, validator, and permission map aligned was sufficient for the API surface.

The core data contract can be represented by this sanitized schema fragment:

```javascript
const ContentConfigSchema = {
  enableDialTest: {
    type: Boolean,
    required: false,
  },
}
```

The actual repository uses Mongoose schema construction and established metadata, but the semantic contract is accurately represented: no migration-created default is required, and false or missing means disabled. I updated the authorization snapshot and ran the survey model and authorization suites. The PR records 22 passing test files and 178 passing tests, including the regenerated permission snapshot. I also applied the local validator, wrote a true value, and read it back to verify that Mongoose and MongoDB accepted the same document shape.

After the API PR merged, I added `enableDialTest` to the survey and setup GraphQL fragments. I selected it unconditionally because GraphQL validation occurs before a feature-flagged component can decide whether to render. If the UI queried the field before the API exposed it, the entire survey-editor query would fail for every user. I documented this deployment dependency explicitly and delayed the UI merge until the swaypi field was available.

The editor control reused the existing content-configuration update path. I also reused the shared `showToggle` policy used by adjacent add-on controls. In simplified form, the visibility logic was:

```javascript
const eligibleVideoPresent = hasNonPlaceboVideo(treatments)
const canConfigure = featureFlags.dialTest && eligibleVideoPresent
const shouldShow = canConfigure || contentConfig.enableDialTest === true
```

The final clause addresses a stranded-state problem. If an author enables Dial Testing and later removes the qualifying video, hiding the control immediately would leave a true setting that the author could no longer see or turn off. The shared factory preserves visibility for an already-enabled control, allowing the configuration to be corrected. This behavior is slightly broader than the initial “hidden when ineligible” statement, but it follows an established product safety pattern and is documented in the merged PR.

I expanded component tests around this behavior. One test asserts that changing the checkbox uses the normal `updateContentConfig` path. Additional tests verify the feature flag and video eligibility gates. I extended the stranded-flag test table and changed the feature-flag mock from one shared reference to a map keyed by flag name. That change allowed the Dial Testing and article-highlighter flags to be controlled independently without coupling otherwise unrelated tests.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

This work demonstrates production programming within an existing architecture. I copied proven schema and UI patterns, coordinated a dependency across two repositories, and made release ordering explicit for reviewers. Rather than designing a novel toggle system, I integrated with the established content-config update path and visibility factory. That decision reduced risk and made the behavior familiar to maintainers.

### 2. Software Architecture and Interface Contracts

I learned to treat a generated GraphQL field as a cross-service contract. The API schema, database validator, authorization map, UI query, and downstream reader all depend on the same name and semantics. I also learned that feature flags do not protect invalid GraphQL selections because query validation occurs before component rendering. This is why the backend deployment had to precede the UI release.

### 3. Human Language Technology and NLP Concepts

Dial Testing produces behavioral time-series data associated with audiovisual media, but my enablement slice did not process that signal and did not implement NLP. There was no speech recognition, language model, sentiment analysis, or text classification in these pull requests. The HLT relevance is contextual: the setting allows a survey to request a richer human-response modality. I describe that relationship without claiming responsibility for the collection mechanism or subsequent analysis.

### 4. Verification and Responsible Feature Delivery

I used model tests, authorization snapshots, a local write/read check, component tests, feature-flag tests, and manual editor verification. I also documented the unreleased backend dependency as a merge blocker. These practices address both correctness and release safety. A small UI control can still make the entire editor unavailable if its query is deployed in the wrong order.

## Challenges

The first challenge was release sequencing. UI PR #2388 selected `enableDialTest` in two GraphQL fragments. Before swaypi PR #1772 was released, that selection was invalid, and the survey-editor query would fail even when the feature flag hid the checkbox. The evidence is recorded in the UI PR dependency warning. I handled this by merging the schema and authorization work first and treating deployment as a prerequisite for the UI.

The second challenge was eligibility state over time. The acceptance criteria focused on showing the toggle when an eligible video exists and hiding it otherwise. Existing shared behavior also kept enabled controls visible after their eligibility condition disappeared. I preserved that behavior to avoid an invisible true flag. The test table now includes the dial-test checkbox, demonstrating that the decision is encoded and regression-protected rather than merely described.

The third challenge was independent feature-flag testing. The prior mock exposed one shared reactive value. Adding another flag would make tests influence each other or falsely represent both flags as having the same state. The updated keyed map provided separate reactive state for each flag. This was a test-infrastructure adjustment inside the touched specification, not a platform-wide feature-flag rewrite.

## Outcomes

The API contract merged on June 24, 2026, and the UI control merged on June 30, 2026. Survey authors with the feature enabled can activate “Collect Dial Test Data” when a non-placebo video treatment is present. The value persists through the normal survey update path and is readable by downstream quiz-building services.

The implementation blocks invalid new configurations, protects already-enabled configurations from becoming invisible, and keeps the schema validator synchronized with the application model. These outcomes enabled the next stages of Dial Testing work. They do not establish that I delivered Dial Testing alone; they establish that I delivered the activation boundary on which the larger feature could depend.

## Professional Practice

I used explicit dependency documentation, cross-repository branch coordination, targeted tests, and a staged release sequence. I followed analogous fields and controls already understood by the team. This is an example of professional restraint: consistency with an established pattern was more valuable than introducing a new abstraction for one Boolean setting.

I also maintain precise attribution. The larger initiative involved product design, respondent components, data collection, and analysis work by other contributors. My portfolio scope is the schema, authorization, GraphQL selection, eligibility and feature-flag gating, persistence control, and associated tests represented in COR-539 and COR-542.

## Code Reference

The code is in private repositories:

- [swaypi #1772 — add `enableDialTest` to survey content configuration](https://github.com/swayable/swaypi/pull/1772)
- [UI #2388 — add the eligible-video-gated editor control](https://github.com/swayable/ui/pull/2388)

The principal implementation areas are the survey Mongoose and JSON schemas, GraphQL authorization permissions, survey and setup query fragments, `SurveySetupContent.vue`, the feature-flag constants, and `SurveySetupContent.spec.js`.
