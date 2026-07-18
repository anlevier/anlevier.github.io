---
title: "Survey Setup Usability Improvements (~25–32 hours)"
excerpt: "<strong>Estimated effort: ~25–32 hours.</strong> Improved survey configuration and respondent usability through custom instructions, visible treatment dependencies, viewport-safe video, and bulk segment-to-filter actions."
collection: portfolio
---

**Estimated effort: ~25–32 hours** (COR-188, COR-227, COR-360, and COR-466)

## Project Overview

I completed four Swayable improvements that addressed friction at different points in survey setup and delivery. Content authors needed to understand and override default respondent instructions. Customer Success users needed to see treatment dependencies without opening every question. Respondents viewing portrait media needed the survey controls to remain reachable. Analysts configuring results needed a faster way to promote several segments into filters.

These tasks did not form one new subsystem, but they shared a usability principle: important state and actions should be visible at the moment a person needs them. Hidden defaults, hidden dependencies, off-screen controls, and repetitive import workflows all increase the chance of configuration errors. I worked within the existing Vue components, stores, API quiz builder, and test patterns to make those states explicit.

For custom instructions, the setup editor displays the default respondent instruction as a placeholder and permits an author-provided override. The quiz-building API renders either the custom value or the established default behavior. For dependency visibility, the question list now shows a distinct treatment-dependency icon and a tooltip that names included or excluded treatments. For media sizing, the video player detects portrait orientation and caps its height to the available viewport. For segments, a bulk action converts selected basic or compound segments into filters while preserving order and avoiding duplicates.

## Technical Approach

The custom-instructions work crossed the setup UI and quiz-building API. I added locale-aware default instruction data, queried the quiz text needed for the current language, and placed the default in the input placeholder. User-entered content continued through the survey setup store. The API changed the final instruction selection so that a custom value takes precedence over the generated default. This allowed authors to preview the distinction between leaving the input untouched and supplying explicit respondent-facing copy.

A sanitized form of the selection rule is:

```javascript
const instructions =
  content.customInstructions !== undefined
    ? content.customInstructions
    : localizedDefaults.contentInstructions
```

The production representation differs, but the ordering is central: use the configured text when supplied, otherwise use the localized default. I verified the respondent preview and documented language-specific QA through the quiz URL.

For treatment dependencies, I added an icon component based on the approved design and integrated it beside the existing question-dependency indicator. The question list inspects included and excluded treatment identifiers, resolves names from setup data, and constructs a tooltip such as “Only shown to [treatment]” or “Hidden from [treatment].” When both dependency types exist, both icons remain visible side by side. The PR deliberately recorded that no automated tests were added because the existing indicator lacked a component-test harness; verification used targeted local data, screenshots, and tooltip QA. I preserve that limitation rather than implying automated coverage.

For portrait media, I followed the viewport calculation already used by other quiz content. The video player applies a maximum height and switches intrinsic sizing according to aspect ratio. Stored dimensions are preferred, while `loadedmetadata` provides a fallback for older video records with no dimensions. A simplified version is:

```javascript
const ratio = storedRatio || naturalVideoWidth / naturalVideoHeight
const sizeClass = ratio < 1 ? "max-width-full height-auto" : "width-full"
const viewportClass = "max-height-available-viewport"
```

The merged PR also applied an image thumbnail clamp. Subsequent evidence on COR-360 links a hotfix that reverted the image clamp to restore article scrolling while retaining the video correction. I therefore claim the durable video viewport behavior and the original investigation of both media types, not that my first image rule remained the final product behavior.

The segment-to-filter action reused the existing bulk-selection toolbar and setup-store write path. For each checked segment, it determines whether a filter already references that segment, skips duplicates, and appends new filters with sort orders after the current maximum. It then clears selection, matching adjacent Combine and Remove actions. Tests cover basic and compound segments, multiple selected items, monotonically increasing order, duplicate avoidance, and selection cleanup.

```javascript
let nextOrder = maximumExistingOrder(filters) + 1
for (const segment of checkedSegments) {
  if (!filters.some((filter) => filter.segmentId === segment.id)) {
    filters.push({ segmentId: segment.id, sortOrder: nextOrder++ })
  }
}
checkedSegments.clear()
```

This sanitized snippet omits proprietary store and schema details while showing the implemented invariants.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

I delivered four reviewed tickets in a mature product repository and one supporting API change. I adapted to different levels of testability, used existing components and stores, preserved unrelated behavior, and documented QA precisely. The work ranged from a one-line API selection change to a stateful bulk operation with unit tests. This variety reflects workplace maintenance and feature development more accurately than a single greenfield assignment.

### 2. User-Centered Software Design

I learned to identify usability problems as mismatches between system state and user visibility. A default instruction that appears only in the respondent flow is hidden state. A treatment dependency visible only inside a side panel is hidden state. A selected segment that requires repeated import actions is visible but operationally expensive. Each solution moved information or action closer to its decision point.

### 3. Human Language Technology and NLP Concepts

The custom-instructions slice handled localized human-facing text, but it did not implement NLP. I worked with language selection and default instructional strings, not language identification, machine translation, parsing, or text generation. The PR documented manual verification with a French locale, but the existing platform supplied localization behavior. My contribution was to retrieve and present the appropriate default and preserve an author override. I make no claim of building a multilingual NLP system.

### 4. Testing and Empirical Evaluation

I used different evidence according to the change. Segment conversion received detailed unit tests because it contained ordering, duplicate, and state-reset logic. Custom instructions used setup tests plus end-to-end preview steps. Dependency icons used controlled fixture data and visual QA, with the absence of automated coverage stated in the PR. Media sizing used before-and-after viewport checks across portrait, landscape, missing-dimension, desktop, and mobile cases. The later image hotfix also showed the importance of evaluating interactions beyond the initial acceptance case.

## Challenges

The custom-instructions requirement contained a data-model constraint: the platform did not store the default instruction as a configured value, so blank historically meant “use default.” COR-188 records the compromise of displaying the default as a placeholder while allowing entered text to override it. The implementation did not redefine every blank-value semantic; it made the existing fallback understandable in the editor and connected custom text to the respondent rendering path.

Treatment dependencies required meaningful tooltips from identifier arrays. PR #2291 documents three QA cases: included treatment, excluded treatment, and a question containing both question and treatment dependencies. It also records that no automated tests covered the pre-existing question-dependency indicator. The evidence supports successful visual and tooltip verification, but not a claim of automated regression coverage for that component.

Media sizing presented a cross-content regression risk. PR #2277 constrained both video and image displays and verified portrait and landscape cases. The later linked hotfix reverted the image clamp because it interfered with article scrolling. This is evidence that a CSS rule that succeeds for one media shape can alter another content interaction. The durable lesson was to apply constraints at the narrowest component level and test scroll behavior, not only final dimensions.

The bulk segment action had to preserve filter order and remain idempotent. PR #2436 records tests for multi-select ordering, basic and compound segments, duplicate skipping, and clearing selection. These tests are direct evidence for the behavior and reduce the risk that repeated use creates duplicate filters or unstable display order.

## Outcomes

The custom instruction UI and API changes merged in April 2026. Authors can see the default instruction while editing and supply content-specific text that reaches the respondent experience. The dependency indicator merged later that month and gives Customer Success users an at-a-glance view of treatment-based routing, including human-readable treatment names.

The portrait video correction keeps tall video within the respondent viewport and uses metadata fallback when stored dimensions are unavailable. The original image constraint was later narrowed by a separate hotfix, so I report that evolution explicitly. The bulk segment-to-filter action merged in July 2026 and changes a repeated one-segment workflow into one selection-and-action operation with duplicate protection.

Together, these changes reduced ambiguity and interaction cost in setup while improving the accessibility of the respondent flow. They are focused improvements within a larger platform, not a claim that I created the full survey editor or quiz experience.

## Professional Practice

I used acceptance criteria, Figma assets, established UI patterns, preview environments, controlled local data, and focused tests. I preserved authorship when continuing the segment action from an earlier spike and documented that history in the pull request. I also recorded testing gaps and later behavior changes rather than presenting every first implementation as final.

The media follow-up is especially important professional evidence. A merged change can reveal an interaction outside the original test matrix. Accurate engineering communication requires acknowledging the correction and narrowing the portfolio claim to the behavior that remained. This practice gives reviewers a more reliable account than a larger but outdated claim.

## Code Reference

The relevant private pull requests are:

- [UI #2234 — custom content instructions](https://github.com/swayable/ui/pull/2234)
- [swaypi #1720 — respondent instruction selection](https://github.com/swayable/swaypi/pull/1720)
- [UI #2291 — treatment dependency indicator](https://github.com/swayable/ui/pull/2291)
- [UI #2277 — viewport constraints for portrait media](https://github.com/swayable/ui/pull/2277)
- [UI #2436 — bulk “Add as Filter” action](https://github.com/swayable/ui/pull/2436)

Representative files include the content details editor, locale constants and quiz query, `TreatmentDependencyIcon.vue`, the survey-question list, `VideoPlayer.vue`, quiz image content, `SurveySetupSegments.vue`, and its unit specification.
