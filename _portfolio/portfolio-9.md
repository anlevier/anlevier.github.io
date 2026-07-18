---
title: "Enterprise Self-Serve and Test Designs (~35–45 hours)"
excerpt: "<strong>Estimated effort: ~35–45 hours.</strong> Developed a coherent client-facing path from understanding a reusable Test Design to configuring, approving, and handing off a survey for launch."
collection: portfolio
---

**Estimated effort: ~35–45 hours** (across COR-32, COR-39, COR-40, COR-269, COR-270, COR-271, COR-300, COR-301, and COR-302)

## Project Overview

During my Swayable internship, I contributed a sequence of changes to the Enterprise Self-Serve initiative. Although the work was divided into several Linear tickets and pull requests, the product problem was unified: clients needed to understand what a Test Design was, create a survey draft from one, configure the draft, and approve it without receiving administrative launch powers.

The existing interface exposed internal concepts such as “Template” and actions that made sense to Customer Success staff but were less clear for clients. I helped turn that interface into a more legible product journey. The home page now labels the collection as “Test Designs,” cards present an explicit “Create from Design” action, and the details and launch panels use consistent language. On the launch page, clients can set a title and description that persist with the draft. The final action distinguishes client approval from immediate launch: a client sees “Approve for Launch,” while an administrator can also select “Launch Immediately + Start Collection.”

This work included larger functional changes and small copy corrections. I treated both as parts of the same information architecture. A single inconsistent label can make users question whether a template, design, and survey draft are different objects. Similarly, an incorrect launch action can communicate authority that the user should not have. The resulting flow gives each screen a more specific purpose: discovery on the home page, context in the details panel, configuration on the launch page, and operational launch control for administrators.

## Technical Approach

I worked primarily in the Vue application, with a supporting schema and authorization change in the Node.js API. I reused shared card components so that the home and Test Designs pages behaved consistently in list and grid modes. The explicit launch action was passed through the existing navigation-item composition rather than implemented independently on each page. I also moved draft-creation behavior into a dedicated composition, which reduced the amount of orchestration embedded in a presentation component.

The client and administrator launch actions were derived from existing feature context. The essential policy can be represented by this sanitized example:

```javascript
const actions = computed(() => {
  const clientAction = { label: "Approve for Launch", kind: "primary" }
  const adminAction = { label: "Launch Immediately + Start Collection", kind: "secondary" }
  return showAdminFeatures.value ? [clientAction, adminAction] : [clientAction]
})
```

The production implementation used established page state and components, but the principle is the same: permissions and role context determine which consequential action is rendered. Cypress coverage exercised the client and administrator paths so the distinction was not left to visual inspection alone.

For the launch-page About section, I added a draft description across the persistence boundary. The API work added a bounded string to the Mongoose model and MongoDB validator, then registered the field with the existing writer authorization template. The UI selected the field through GraphQL and bound title and description edits to the draft save path. This required the model, database validator, GraphQL permissions, query, form, and launch conversion to agree. A missing read permission initially caused the entire draft query to fail for editor users; granting the field through the standard writer template restored the expected end-to-end behavior.

The remaining Test Design changes reinforced the conceptual model. I renamed navigation card labels, updated the “Created from” card on launch and status pages, removed an early Preview link that did not represent the configured survey, and changed “Launch Design” to “Launch from Design.” These were intentionally narrow changes, often with updated Cypress assertions or targeted QA. I did not inflate their technical scope, but I considered their cumulative product effect.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

This project most directly demonstrates workplace programming. I delivered work through nine tracked tickets and a series of reviewed pull requests in two repositories. I followed existing Vue composition, GraphQL authorization, Mongoose, JSON-schema, and Cypress conventions. I also learned to separate product policy from presentation details. The role-sensitive launch action was not merely different button text; it enforced the intended handoff between a client and Customer Success.

### 2. Software Design and System Integration

I practiced tracing a user-facing field through a distributed application. The draft description had to be represented consistently in the browser, GraphQL query, authorization map, Mongoose schema, MongoDB validator, and survey-creation path. The experience reinforced that a form is complete only when data can be written, reloaded, authorized, and transferred into the downstream entity.

### 3. Human Language Technology and NLP Concepts

This project did not implement an NLP algorithm, corpus, model, tokenizer, or linguistic analysis pipeline. Its relationship to human language technology was limited to terminology, instructional clarity, and interface copy in a research platform. I applied careful language choices to reduce conceptual ambiguity, but I do not claim that copy changes constitute NLP work. The strongest learning outcome here is product communication around a technology-enabled workflow.

### 4. Testing, Documentation, and Technical Communication

I used pull-request descriptions, preview deployments, screenshots, Cypress tests, and explicit QA steps to communicate behavior to reviewers. Tests covered role-dependent launch actions and launch-from-draft behavior. For small copy-only changes, I documented the limited risk and verified the affected surface instead of claiming broad automated coverage. This proportional approach made the evidence for each change visible.

## Challenges

The main challenge was maintaining one product vocabulary across components that reused older internal names. The evidence is distributed across the merged work: COR-269 changed the home heading, COR-270 changed navigation card labels, COR-300 aligned the “Created from” card on two pages, and COR-302 aligned the side-panel title. Because each change touched a different surface, repository-wide searches and page-specific QA were necessary to detect stale language.

A second challenge was the authorization failure exposed by the About section. The UI requested the new description field, but editor users initially received a GraphQL “Not Permitted” error. The failed query prevented other draft data, including the organization identifier, from being populated, and a later save surfaced as a generic failure. The merged API PR records the resolution: the description field was added to the standard writer permissions and the UI end-to-end tests then passed against the backend branch.

The launch controls also required evidence across distinct user states. PR #2143 changed the page and expanded the Cypress launch-from-draft specification. The documented behavior was that administrators saw both actions, while clients without administrative features saw only approval. I therefore describe the implemented conditional behavior, not a broader authorization redesign.

## Outcomes

All referenced changes were merged. The delivered path gives clients a direct entry point from a Test Design card, consistent language through the details and launch views, persisted title and description fields, and an approval action that matches their role. Administrators retain the immediate launch action required for operational work.

The project also reduced misleading or unnecessary interface elements. Removing the premature Survey Instrument preview focused the details panel on design information, while the later launch view remained the appropriate place to preview a configured test. Empty ellipsis menus were removed for non-administrators after the primary “Create from Design” action moved onto the card. These outcomes made the flow more understandable without claiming that I designed or built the entire Enterprise Self-Serve product.

## Professional Practice

I worked from Figma specifications, Linear acceptance criteria, repository conventions, review feedback, and staged preview environments. I kept cross-repository branches aligned where UI and API changes had to be tested together. I recorded exact QA paths and distinguished automated checks from manual verification.

This work also taught me to evaluate scope at the level of a user journey. Several tickets were only one-line copy changes, while others added persistence or role-aware behavior. Reporting them as one coherent story is accurate because they converged on the same workflow, but professional attribution requires preserving the boundaries of my contribution. I implemented these slices within a larger initiative owned by a multidisciplinary team.

## Code Reference

The source repositories are private. Authorized viewers can inspect the principal pull requests:

- [UI #2143 — role-appropriate launch buttons](https://github.com/swayable/ui/pull/2143)
- [UI #2164 — Test Design card options](https://github.com/swayable/ui/pull/2164)
- [UI #2156 — launch About section](https://github.com/swayable/ui/pull/2156)
- [swaypi #1704 — SurveyDraft description schema and permissions](https://github.com/swayable/swaypi/pull/1704)
- [UI #2228, #2239, #2268 — Test Design naming and creation path](https://github.com/swayable/ui/pull/2239)
- [UI #2258, #2254, #2253 — launch and details refinements](https://github.com/swayable/ui/pull/2258)

Representative implementation areas include `pages/launch/[surveyDraft_id].vue`, shared Test Design card components, `NavigationItem.vue`, the draft-creation composition, the launch Cypress specification, and the `SurveyDraft` schema and permission definitions.
