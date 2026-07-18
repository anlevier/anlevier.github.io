---
title: "Developer Environment and Onboarding (~20–28 hours)"
excerpt: "<strong>Estimated effort: ~20–28 hours.</strong> Improved Swayable onboarding through Docker runtime alignment, environment examples, service documentation, and diagnosis of schema-dependent Cypress setup failures.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

**Estimated effort: ~20–28 hours**

## Project Overview

My first productive engineering work at Swayable depended on making a multi-service local environment understandable and repeatable. The relevant record includes onboarding tickets ENG-1054 and ENG-1055, documentation tickets ENG-1082 and ENG-1120, retrospective items RET-174, RET-214, and RET-287, merged pull requests sway #8, swaypi #1637, and swayable-data #2727, and the later diagnostic pull request sway #49. Together, these items show a progression from setting up individual services to improving the shared instructions and then reasoning about failures that appeared only when Docker, database schemas, and Cypress interacted.

Swayable’s development environment spans a Node.js API, a Python data service, MongoDB, Redis, background workers, user-interface services, and Docker orchestration. Onboarding was therefore not a matter of installing one package and running one command. Environment variables had to point to compatible local services, container images had to use supported runtimes, and startup order mattered. RET-174, for example, records a concrete documentation problem: Celery required Redis, but the Redis instructions appeared after the Celery steps. RET-214 records collaborative help with the local environment, and RET-287 records successful local Docker testing.

The goal of this theme was not to redesign the development platform. My contribution was to reduce specific sources of setup ambiguity, align one container runtime, and document configuration that the services already expected. I also diagnosed a schema-upgrade gap in the end-to-end test path. Because sway #49 was closed without merge, I treat it as evidence of technical diagnosis and learning, not as a shipped change in the `sway` repository.

## Technical Approach

### Learning the service graph through setup

ENG-1054 and ENG-1055 tracked setup of `swaypi` and `swayable-data`. Completing these tasks required understanding which processes were independent and which were prerequisites. The API depended on MongoDB and environment-specific URLs. The Python service used Flask configuration and could dispatch background tasks that required a broker. Docker provided repeatable infrastructure, but the application processes still depended on valid `.env` values and compatible runtime versions.

I treated setup errors as information about architecture. A connection failure could indicate a missing container, an incorrect host name, a wrong port, or startup order. A worker failure could indicate that Redis was not running rather than a Python defect. This approach prevented me from adding arbitrary configuration until a command happened to work. Instead, I traced each service to its documented dependency and then updated documentation where the observed dependency was absent or out of order.

### Making environment examples executable documentation

In swaypi #1637, corresponding to ENG-1082, I updated `.env.example` and the README with logging options, MongoDB connection alternatives, application URLs, and the API port variable. The pull request added 30 lines and removed four from the example file, plus a small README update. The important outcome was not the line count. An environment example acts as an interface between a repository and a new developer, so names and grouping must match the application’s actual configuration surface.

In swayable-data #2727, corresponding to ENG-1120, I added eight Flask-related variables to `.env.example`. This corrected a mismatch where configuration was described in documentation but not represented in the file developers copy when constructing a local environment. Keeping these sources aligned reduced the chance that a developer would follow the README yet still start the service without required values.

I did not place credentials or production values in these examples. The work documented variable names and safe local structure, preserving the distinction between configuration guidance and secret distribution.

### Aligning the Docker runtime

Sway #8 updated the Node.js version in three Docker Compose files: the install, test, and standard development definitions. The change was three additions and three deletions and merged in February 2026. Applying the version consistently mattered because different Compose entry points should not silently test or install under a different runtime from the one used for normal development.

This task reinforced the principle that runtime declarations are part of the build contract. A developer can have the correct Node.js version on the host and still experience inconsistent behavior if the container definition pins another version. Updating all three Compose paths reduced that category of drift.

### Diagnosing schema-dependent Cypress failures

The most technically instructive environment issue appeared later in sway #49. Cypress setup tests attempted to write a newly introduced field. The Mongoose model and branch-level JSON schema included that field, but the preloaded MongoDB image used by continuous integration carried validators built from the main branch. Because those validators rejected additional properties, the UI-to-API-to-database path failed with document validation even though unit tests were green.

I identified `schema:upgrade` as the operation that synchronized live validators with repository schemas and proposed running it after the end-to-end stack started. The proposal was idempotent and targeted the gap between a preloaded database image and branch-specific schema changes. However, pull request #49 was closed without merge. I therefore do not claim that I shipped this workflow change. Its portfolio value is the diagnosis: the failure was not fundamentally a Cypress selector problem or an application model problem; it was version skew between a branch and the validator state inside test infrastructure.

That diagnosis also informed later sampling work. It explained why a user-facing field could pass unit tests yet fail only when Cypress exercised a real database write. I reference that reliability lesson under sampling and environment practice, while preserving the distinction between proposed and merged work.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

I developed practical skill in configuring and testing polyglot services. I used repository conventions, made small reviewable pull requests, and updated examples alongside documentation. I learned to verify a change through the same execution path a teammate would use rather than assuming that syntactically valid configuration was sufficient.

### 2. Fundamental NLP and Data-System Concepts

Although this theme did not implement an NLP algorithm, it established the operational foundation for running analysis services. Background language-processing and statistical tasks depend on a broker, worker processes, database connectivity, and reproducible package environments. The schema diagnosis also demonstrated a core data-engineering concept: application models and database validators are separate layers that must evolve together.

### 3. Tools and Packages

The work involved Docker Compose, Node.js, Yarn, Python, Flask, Celery, Redis, MongoDB, Mongoose models, JSON Schema validators, Cypress, `.env` conventions, local logging configuration, and GitHub Actions-style continuous-integration flows. I used these tools as an integrated system rather than as isolated technologies.

### 4. Workplace Communication and Collaboration

Onboarding required asking focused questions and converting answers into reusable documentation. RET-214 records collaborative local-development help, while RET-174 records a precise ordering problem in the instructions. I learned to report setup friction as a reproducible dependency issue rather than as a general statement that the environment did not work. That level of specificity makes documentation feedback actionable.

## Challenges

The evidence supports three specific challenges. First, the service dependency order was not fully reflected in onboarding instructions. RET-174 states that Celery required Redis while the Redis steps appeared later. This supports a documentation-ordering claim, but it does not support a claim that all onboarding documentation was incomplete.

Second, configuration information was split between prose and example files. ENG-1082 and ENG-1120, together with merged pull requests #1637 and #2727, show exactly which categories were added: logging, MongoDB, URLs, a service port, and Flask variables. These merged changes support improved discoverability for those settings.

Third, branch-specific database schema changes were incompatible with validators in a preloaded test image until an upgrade step ran. Pull request #49 documents the observed `additionalProperties` validation failure and the proposed idempotent upgrade. Since the pull request was closed without merge, the evidence supports diagnosis and a proposed remedy only. It does not support describing the schema upgrade as part of the shipped `sway` test workflow.

## Outcomes

Three changes shipped: consistent Node.js container versions across three Compose definitions, an expanded `swaypi` environment example and README, and Flask variables in the `swayable-data` environment example. The setup tickets for both services were completed, and the retrospective record later documented successful local testing with Docker.

The broader outcome was a more accurate mental model of the system. I learned how local API, data, database, broker, worker, and browser-test layers fit together. That understanding supported later feature and reliability work because I could reproduce production-like state locally, interpret failures at the correct layer, and write QA steps that other engineers could follow.

I cannot quantify a reduction in onboarding time from the available evidence, and I do not present the closed schema proposal as deployed. The documented outcome is narrower: specific setup information was added, a runtime mismatch was corrected, local Docker testing succeeded, and a difficult end-to-end schema failure received an evidence-based root-cause analysis.

## Professional Practice

This work taught me that onboarding is an engineering activity. A new developer follows interfaces that the team maintains: Compose files, example environments, READMEs, task recipes, and database bootstrap behavior. If those interfaces disagree, the developer bears the integration cost. Improving them creates leverage beyond one person, even when the code diff is small.

I also learned to preserve status distinctions in technical reporting. A merged environment change can be described as an outcome. A closed pull request can be described as investigation, diagnosis, or a proposal, but not as shipped behavior. This distinction is important in a professional portfolio because an honest account of a well-supported diagnosis is stronger than an inflated implementation claim.

## Code Reference

The repositories are private. Authorized viewers can review:

- [sway #8 — align Node.js runtime in Docker Compose](https://github.com/swayable/sway/pull/8)
- [swaypi #1637 — update README and environment example](https://github.com/swayable/swaypi/pull/1637)
- [swayable-data #2727 — add Flask variables to environment example](https://github.com/swayable/swayable-data/pull/2727)
- [sway #49 — schema upgrade before Cypress, closed without merge](https://github.com/swayable/sway/pull/49)

The environment reasoning can be summarized with sanitized pseudocode:

```text
start(database, broker)
apply(current_branch_database_schemas)
start(api, data_service, workers)
run(browser_tests)
```

This sequence is conceptual. The proposed automatic schema step in sway #49 was not merged.
