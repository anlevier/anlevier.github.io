---
title: "Swayable Internship Overview (~420–545 hours across themes)"
excerpt: "<strong>Estimated effort: ~420–545 hours across themes.</strong> A canonical overview of my Swayable software engineering internship, connecting production NLP, full-stack product delivery, reliability, research prototypes, developer enablement, and customer operations.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

**Estimated effort: ~420–545 hours across themes**

## Project Overview

I completed my internship at Swayable, a research-technology company whose platform measures persuasion through online surveys and combines survey operations, statistical analysis, qualitative processing, and client-facing results tools. I worked as a software engineering intern within Swayable’s engineering environment. My work crossed private production repositories for the Vue user interface, the Node.js API, the Python data and analysis service, and shared Docker-based development infrastructure.

This page is the canonical summary of the internship. The linked portfolio entries describe individual themes in greater technical detail and preserve evidence boundaries around merged code, prototypes, operational tickets, and collaborative work. The combined estimate is presented as approximately 420–545 hours. Individual theme ranges are evidence-based estimates rather than a timesheet and contain some natural overlap where one debugging or integration activity supported more than one theme. The overall range is therefore rounded and should not be interpreted as the exact arithmetic sum of independently billable projects.

My role developed from onboarding and focused changes into ownership of larger cross-service features and production investigations. Early work included service setup, environment documentation, and a small AI feature-flag removal. Later work included automated sentiment and relevance processing, a substantial substantiveness-classification pipeline, end-to-end survey features, an information-extraction prototype, production reliability, and time-sensitive operational support.

## Internship Setting, Role, and Supervision

The internship was carried out with Swayable in its normal software-development workflow. Work was planned and tracked in Linear, implemented through Git branches and GitHub pull requests, tested locally and in continuous integration, and reviewed by Swayable engineers. Preview environments, staging data, logs, unit tests, Cypress tests, and documented quality-assurance paths provided evidence for behavior before merge.

My supervision took the form of ticket definition, engineering review, collaborative debugging, retrospectives, and feedback on pull requests. Senior and peer reviewers challenged configuration defaults, test coverage, architectural boundaries, feature rollout, user-state behavior, and appropriate repository placement. I do not include supervisor quotations because no approved quotation is part of the evidence for these entries.

I worked within a larger multidisciplinary organization. Product requirements and designs came from shared planning, while engineers, Customer Success staff, and support teammates contributed context about user workflows and operational urgency. I distinguish my implementation from dependencies owned by others. For example, the Article Highlighter entry identifies the configuration, schema, quiz-generation, and store layers that I implemented while excluding a persistence mutation and presentation work completed by teammates.

## Goals and Methods

My internship goals were to apply graduate study in Human Language Technology to a production organization, strengthen workplace programming practices, learn the operation of a mature multi-service platform, and contribute useful software. Those goals required both NLP-specific and general engineering work.

For language-processing projects, I used staged classification, prompt templates, asynchronous task queues, persisted processing state, and structured outputs. I learned that an NLP model is only one component of a production system. Selection, gating, retries, idempotence, observability, cost controls, and persistence determine whether model output becomes dependable product data.

For full-stack features, I traced values and permissions from Vue components through GraphQL, authorization, Mongoose models, MongoDB validators, conversion commands, and respondent experiences. This method was especially important for sampling instructions, Article Highlighter configuration, Dial Testing activation, and launch-page data.

For reliability and support, I began with an observable acceptance condition, inspected state at the implicated layer, and avoided claims beyond available evidence. I used regression tests when I changed code and documented honest limits when work consisted of diagnosis or operations rather than an authored pull request.

For experimental work, I treated evaluation and architectural feedback as outcomes. The autoprogram parser was a reviewed proof of concept that did not merge. Its value was an executable extraction pipeline, a comparison harness, and evidence that a brittle regex approach should give way to structured LLM output in an isolated service boundary.

## Portfolio Themes

The internship is organized into the following theme pages:

1. [Substantiveness Classification Pipeline — ~120–150 hours](/portfolio/portfolio-4/). I built a multi-stage LLM pipeline for question intent, response type, and substantiveness classification with schema, Celery, API, prompt, and batch-runner work.
2. [Sentiment and Relevance Automation — ~35–45 hours](/portfolio/portfolio-5/). I hardened inherited classifier work into a scheduled, observable, retryable Celery pipeline.
3. [Autoprogram: Google Doc Question Extraction — ~40–50 hours](/portfolio/portfolio-6/). I developed and evaluated an unmerged information-extraction prototype and documented why the next architecture should change.
4. [Article Highlighter Collection — ~45–55 hours](/portfolio/portfolio-7/). I implemented major full-stack layers for collecting structured human text-span annotations.
5. [Launch-Page Autosave — ~12–16 hours](/portfolio/portfolio-8/). I built concurrency-safe persistence and unified save-state feedback.
6. [Enterprise Self-Serve and Test Designs — ~35–45 hours](/portfolio/portfolio-9/). I contributed role-appropriate launch behavior, durable draft fields, and consistent product language.
7. [Sampling Information, End to End — ~15–20 hours](/portfolio/portfolio-10/). I carried a human-authored sampling field across schema, permissions, launch, setup, and end-to-end tests.
8. [Dial Testing Activation — ~10–14 hours](/portfolio/portfolio-11/). I delivered the schema and eligibility-gated editor control for a larger behavioral-data feature.
9. [Survey Setup Usability Improvements — ~25–32 hours](/portfolio/portfolio-12/). I improved instructions, dependency visibility, video sizing, and bulk segment workflows.
10. [Production Reliability and Debugging — ~25–32 hours](/portfolio/portfolio-13/). I corrected lifecycle timing, automation email delivery, null-data handling, and organization-context lookup.
11. [Developer Environment and Onboarding — ~20–28 hours](/portfolio/portfolio-14/). I improved Docker runtime consistency, environment examples, service documentation, and schema-failure diagnosis.
12. [Customer Support and Production Operations — ~30–40 hours](/portfolio/portfolio-15/). I handled analysis, configuration, access, metric, and segmentation requests with evidence-based operational practice.
13. [AI Feature Rollout and Access — ~5–8 hours](/portfolio/portfolio-16/). I removed a redundant AI Toplines flag and supported access verification while accurately limiting the claim to presentation and rollout.

Team Enablement was considered as a possible additional theme but was intentionally not written as a standalone portfolio entry. COR-43, COR-77, and ENG-1016 are supporting context for collaboration and internal enablement. Their learning is incorporated into this overview and the professional-practice sections of the technical themes.

## Results and Contributions

The most substantial NLP outcome was a production-oriented classification system for open-ended survey responses. The substantiveness project introduced typed qualitative dimensions and a conditional multi-stage path for applying them. The sentiment and relevance project established safe periodic execution with durable retry semantics. Together, these projects connected NLP concepts to production concerns such as cost, queue behavior, schema evolution, and observable completion.

The full-stack work delivered several user-facing capabilities. Survey authors gained configuration paths for article highlighting, sampling information, Dial Testing activation, custom instructions, and launch workflow improvements. Respondent and analyst experiences gained viewport-safe media, structured highlight data, visible dependencies, and bulk configuration actions. I report these as scoped contributions within a larger platform rather than claiming ownership of the platform or complete product initiatives.

The reliability work corrected concrete failure modes across services: a notification tied to the wrong lifecycle event, real email delivery to automation users, a null dereference caused by deleted child data, and an undefined organization identifier that blocked a client results workflow. These changes taught me that small diffs can carry significant diagnostic and operational value.

The internship also produced non-code outcomes. I improved environment examples, learned a Docker-based service graph, documented setup dependencies, supported urgent production work, and developed a repeatable diagnostic method. The autoprogram prototype provided negative as well as positive knowledge: it showed that an evaluation harness was valuable while a growing rule-based parser was not the appropriate production direction.

I do not claim unsupported business metrics, model-accuracy improvements, support-volume reductions, or supervisor endorsements. Where no authored pull request exists, the relevant theme is framed as operations and reflection. Where a pull request closed unmerged, it is labeled as a prototype, diagnosis, or proposal.

## Skills and MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

I gained experience delivering software in mature repositories with existing conventions. I wrote Python, JavaScript, and Vue code; coordinated cross-repository changes; responded to review; documented QA; and used automated testing at unit, component, schema, and browser levels. I learned to prefer a narrow change that protects a stated invariant over a broad rewrite that is harder to verify.

### 2. Human Language Technology Theory and Practice

The internship connected HLT theory to practical systems. Text classification appeared as zero-shot LLM prompting, cascaded gating, sentiment, relevance, response-type, and substantiveness labels. Information extraction appeared in the autoprogram prototype through segmentation, normalization, weak structural cues, schema mapping, and approximate matching. Human annotation appeared in Article Highlighter spans and comments.

The strongest theory-to-practice lesson was that model behavior and system behavior are inseparable in production. A classifier may be conceptually sound but operationally unreliable if tasks duplicate, failures disappear, irrelevant records consume model calls, or completion is announced too early. Conversely, a feature can support language technology without training a model, as when software reliably collects text spans, preserves human-authored instructions, or presents generated summaries only when valid output exists.

### 3. Tools, Data, and Infrastructure

I worked with Vue, GraphQL, Node.js, Python, Flask, Celery, MongoDB, Mongoose, JSON Schema, Jinja2, Docker Compose, Redis, Cypress, unit-test frameworks, environment configuration, and private GitHub workflows. More importantly, I learned how these tools form one system. A branch-level model change can fail against a stale database validator. A hidden Vue component can still break a query if GraphQL validates an unreleased field. A task queue requires durable state and recovery policy, not only a decorated function.

### 4. Professional Communication and Judgment

I learned to communicate status precisely: merged, unmerged, diagnosed, proposed, tested, or operationally completed. I practiced sanitizing private information, separating user impact from root cause, identifying collaborative ownership, and recording testing limits. Review and support work also strengthened prioritization under uncertainty. I learned that urgency increases the need for concise evidence; it does not reduce the need for it.

## Professional Practice and Reflection

Across the internship, I moved from understanding individual services to reasoning about boundaries among systems, teams, and product states. I became more comfortable entering an unfamiliar code path, reconstructing its invariant, and choosing evidence that could prove a correction. I also became more willing to remove code, close an unsuitable implementation path, or describe a small contribution honestly.

The internship confirmed that Human Language Technology work in industry includes much more than model training. Production HLT depends on software architecture, data contracts, asynchronous execution, interfaces, monitoring, access, and support. My strongest contributions combined language-technology understanding with disciplined software engineering.

This overview and its theme structure may be adjusted if my academic advisor requests a different internship-report format. Pending that format approval, this page serves as the canonical summary, and the linked entries provide the detailed evidence.

## Code Reference

All Swayable source repositories and issue records referenced by these pages are private. Individual theme pages link authorized viewers to the relevant pull requests and provide sanitized pseudocode or descriptions. No credentials, customer data, private record identifiers, or proprietary source excerpts are reproduced in this portfolio.
