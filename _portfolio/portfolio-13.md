---
title: "Production Reliability and Debugging (~25–32 hours)"
excerpt: "<strong>Estimated effort: ~25–32 hours.</strong> Diagnosed lifecycle, data-integrity, automation, and organization-context failures across Swayable services, then implemented focused safeguards and regression coverage.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

**Estimated effort: ~25–32 hours**

## Project Overview

During my Swayable internship, I completed a group of production-reliability tasks spanning the Python analysis service, the Node.js API, and the Vue user interface. The work is represented by tickets COR-46, COR-61, COR-532, ENG-2313, and ENG-2446 and by four merged pull requests: swayable-data #3023, swaypi #1714 and #1765, and ui #2211. Although each correction was small relative to a feature project, the group required careful reasoning about system lifecycle, incomplete data, automated test behavior, and state ownership across service boundaries.

The central theme was that a locally reasonable assumption can become a production failure when the surrounding lifecycle is more complicated. A “finalize” click does not mean analysis has finished. A database reference does not guarantee that the referenced child still exists. A test account is not harmless if production email infrastructure attempts delivery. A user session store does not necessarily own the active organization identifier. Reliability work required me to locate these mismatches, identify the true source of state, and make the narrowest change that restored the intended invariant.

One task was directly client-blocking. In ENG-2446, the multi-test explorer displayed no candidate tests, preventing a requested combined results view. The issue record gave a three-hour urgency and stated that results delivery was blocked. I traced the empty state to a GraphQL variable populated from a nonexistent property on the authentication store. The organization identifier actually belonged to the organization store. A three-addition, one-deletion UI change restored the query context, and ui #2211 merged on March 19, 2026.

## Technical Approach

### Aligning notifications with completed analysis

COR-46 concerned the “results are ready” email. The prior behavior associated notification delivery with the administrative finalize action, even though final analysis could still be running. I followed the lifecycle beyond the initiating UI event and located the state transition that records final analysis as complete. In swayable-data #3023, I moved notification eligibility to the point where analysis metadata is marked final. This changed the invariant from “the user requested finalization” to “the system has completed final analysis.”

The production edit was only a few lines in the analysis task, but I added substantial regression coverage. The pull request changed two production lines and added 121 test lines. The tests represented pending and successful workflow states and verified that notification behavior followed computed analysis status. This ratio was appropriate because timing and state-transition bugs are easy to reintroduce if only the final happy path is checked.

### Protecting production services from automation

COR-61 showed that Cypress-generated users were reaching the real email relay. Those addresses bounced, creating unnecessary delivery traffic and potentially damaging sender reputation. I placed the safeguard in the shared SMTP relay rather than in individual welcome, password-reset, or account flows. Centralizing the rule meant every current caller used the same protection and future callers would inherit it.

In swaypi #1714, the relay routes recognized Cypress recipients to logging instead of the real transporter. I added unit coverage for multiple email types and corrected an adjacent defect where a credential-checking function had been referenced without being called. The pull request also included a small command-line test utility so a developer could compare normal and Cypress-recipient behavior. The merged change therefore combined a production guard, regression tests, and an operator-facing verification path.

### Guarding partially deleted data

COR-532 surfaced during a staging backfill when `getThumbnailImage` encountered a multiple-content item whose first child was absent. The issue evidence documented two failure paths: an empty asset array and a reference to deleted or missing content. In either case, code later attempted to read `content.type` from an invalid value.

In swaypi #1765, I used optional chaining for the child identifier and placed a null guard after the nested lookup. The guard’s location matters. An earlier check protected the outer content lookup, but a second database query inside the multiple-content branch could replace a valid parent object with `null`. Checking after that branch covers both paths before the type switch. I also added a regression test for the missing-child case. This was a defensive change, not an attempt to reconstruct deleted data; when no valid thumbnail source exists, returning without a thumbnail is safer than crashing a save hook or backfill.

### Restoring organization context

For ENG-2446, I reproduced the empty multi-test explorer and followed the data path from the Vue component to its GraphQL query. The component read `organizationId` from the authentication store, but that property did not exist there. The organization store exposed the active value as `organization_id`. In ui #2211, I imported the appropriate store and supplied its value to the query.

This correction illustrates state-ownership debugging. The visible symptom was an empty search result, but the underlying issue was neither search nor authorization. The query was valid syntactically but carried an undefined organization context. Testing therefore included the reported workflow and consideration of a non-admin account so that a context fix was not mistaken for a permissions fix.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

I practiced reading unfamiliar code paths across Python, Node.js, and Vue rather than treating repository boundaries as debugging boundaries. I made focused changes, wrote regression tests proportional to risk, documented reproducible QA steps, and used Linear and GitHub to preserve the reasoning behind each correction. The work reinforced that maintainable production fixes should state the invariant they protect, not merely suppress the observed symptom.

### 2. Fundamental NLP and Data-System Concepts

These tasks were not model-development projects, but they supported the reliability of a platform whose analysis pipeline includes statistical and language-processing stages. COR-46 required distinguishing an asynchronous request from completion of the underlying computation. COR-532 required handling missing references in a document database. Both concepts matter in production NLP systems, where downstream summaries or classifications should not be announced until processing is complete and where historical records can contain incomplete relationships.

### 3. Tools and Packages

I used Python task code and tests in `swayable-data`, Node.js email and database modules in `swaypi`, Vue and Pinia-style stores in `ui`, MongoDB fixtures for lifecycle reproduction, GraphQL query inspection, Vitest-based unit tests, Cypress account conventions, Docker-based local services, and GitHub pull-request review. I also used logs and explicit database state setup to distinguish event timing from computed completion.

### 4. Workplace Communication and Problem Solving

The ticket set required translating reports such as an empty selector or an early email into testable technical hypotheses. I recorded reproduction steps, constrained claims to evidence, and explained why each fix belonged at a particular abstraction boundary. The client-blocking issue also required prioritization: I concentrated first on restoring the missing organization context, while still validating that the correction behaved under the relevant account context.

## Challenges

The first evidence-supported challenge was lifecycle ambiguity. COR-46 explicitly documented that the email was sent when finalization was clicked instead of when final analysis completed. The corresponding pull request moved the trigger to analysis completion and added tests around analysis metadata. I do not claim a broader notification redesign; the evidence supports a targeted correction to one lifecycle boundary.

The second challenge was a latent null path revealed by batch execution. COR-532 records the exact null-pointer location, the nested lookup that could return no child, and the fact that a backfill exercised the path across many surveys. The resulting pull request added optional chaining, a post-lookup guard, and a regression test. The evidence supports improved tolerance for empty or deleted child assets, but it does not establish that every form of historical content inconsistency was repaired.

The third challenge was separating an empty-result symptom from its cause. ENG-2446 documented a client-blocking empty multi-test explorer. Pull request #2211 identified an undefined organization variable caused by reading the wrong store and changed only that context source. The evidence supports restoration of candidate lookup for the reported path, not a general rewrite of multi-test search.

Finally, COR-61 required treating automated tests as participants in production-adjacent infrastructure. The ticket and pull request establish that Cypress recipients were sent through the real relay and bounced. The merged guard and tests support the claim that recognized Cypress recipients are logged instead of delivered through that shared relay.

## Outcomes

All four referenced pull requests merged. Results-ready emails now align with completed final analysis rather than the initiating click. Cypress recipient addresses are intercepted at the central relay. Thumbnail selection tolerates an empty or deleted multiple-content child without dereferencing `null`. The multi-test explorer receives organization context from the store that owns it.

Collectively, these outcomes reduced four distinct reliability risks: premature communication, automation side effects, crashes on incomplete historical data, and a client-blocking UI failure. The work also produced regression coverage for the notification, email, and deleted-child paths. I describe the impact at this level because the repositories and operational telemetry are private; I cannot substantiate numerical reductions in incident rate or support volume.

## Professional Practice

This theme changed how I approach production debugging. I learned to identify the system invariant before editing code: notify only after completion, never deliver to recognized test recipients, never dereference an unverified database lookup, and source organization context from the authoritative store. An invariant gives reviewers a stable reason for the patch and helps determine what the tests should prove.

I also learned that small diffs can represent substantial engineering work. Pull request #2211 changed four lines, while the diagnosis crossed component state, GraphQL variables, and account context. Pull request #3023 changed only a few production lines, but lifecycle reconstruction and test design accounted for most of the effort. In professional practice, line count is not an adequate measure of complexity or value.

## Code Reference

The implementation is in private Swayable repositories and is available to authorized viewers:

- [swayable-data #3023 — send results email after final analysis](https://github.com/swayable/swayable-data/pull/3023)
- [swaypi #1714 — skip delivery to Cypress users](https://github.com/swayable/swaypi/pull/1714)
- [swaypi #1765 — guard missing thumbnail child](https://github.com/swayable/swaypi/pull/1765)
- [ui #2211 — use organization context in multi-test explorer](https://github.com/swayable/ui/pull/2211)

Representative sanitized logic is:

```text
if final_analysis_completed:
    queue_results_notification()

if recipient_is_automation_user:
    log_message_instead_of_delivering()

child = lookup(first_asset_id_if_present)
if child is missing:
    return no_thumbnail

organization_id = organization_store.active_id
query_candidates(organization_id)
```
