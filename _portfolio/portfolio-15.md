---
title: "Customer Support and Production Operations (~30–40 hours)"
excerpt: "<strong>Estimated effort: ~30–40 hours.</strong> Investigated urgent production-analysis, configuration, access, metric, and segmentation requests while communicating evidence, impact, and resolution status across teams.<br/><img src='/images/500x300.png'>"
collection: portfolio
---

**Estimated effort: ~30–40 hours**

## Project Overview

Customer support and production operations formed a distinct part of my Swayable internship. The work is represented by TSUP-77, TSUP-85, TSUP-87, TSUP-92, TSUP-93, TSUP-162, ENG-2436, ENG-2449, ENG-2451, ENG-2514, and ENG-2524. I did not author linked pull requests for this group. The evidence instead consists of completed support records, urgency and impact statements, issue timelines, and acceptance criteria. I therefore frame this entry as operational impact and professional reflection rather than as a software feature.

The tickets covered several recurring categories: analysis or reanalysis that appeared stalled or failed, configuration validation, a missing calculated metric, client-user access, and assignment of respondents to defined segments. Many records explicitly stated that results delivery was blocked or that a service-level expectation was at risk. The practical goal was to move each request from an ambiguous user-visible symptom to a verified technical state, then communicate either a resolution or a defensible next action.

The work also showed how operational support differs from planned development. A feature ticket usually provides time for design, implementation, automated tests, and review. A production request often begins with an urgent report, limited context, and a deadline. I still needed to preserve accuracy. Urgency could change prioritization, but it could not justify guessing about data, rerunning destructive actions without understanding them, or claiming a root cause that the record did not establish.

## Technical Approach

I used a layered diagnostic process. First, I translated the request into an observable acceptance condition. For analysis incidents, that condition was not merely that a button had been clicked; it was that analysis completed and the requested results were available. For validation incidents, it was that the affected configuration no longer produced the documented error. For user-access incidents, it was that the intended account path functioned. For segmentation work, it was that the requested segment assignments were present.

Second, I identified the layer most directly implicated by available evidence: user interface, stored configuration, asynchronous task state, calculated output, account state, or data assignment. I compared expected and observed state before taking action. This reduced the risk of repeatedly triggering a workflow when the underlying issue was configuration or of editing configuration when a background job was simply still active.

Third, I maintained a boundary between diagnosis and escalation. If the available state could verify a known operational path, I followed that path and confirmed the acceptance condition. If evidence pointed to a code defect or a system state outside the safe support scope, the correct action was to preserve the observations and route the issue to the appropriate engineering owner.

A sanitized representation of this process is:

```text
receive_request()
remove_client_identifiers_from_notes()

expected = define_acceptance_condition(ticket)
observed = inspect_visible_state_and_diagnostics()

if observed indicates active work:
    verify_progress_without_duplicate_trigger()
elif observed indicates known configuration gap:
    apply_authorized_configuration_correction()
elif observed indicates failed or inconsistent system state:
    collect timestamps, task state, and reproducible symptoms
    use approved recovery path or escalate

verify(expected)
record_only_evidence_supported_outcome()
```

This is process pseudocode, not source code from a specific incident. It intentionally omits client identifiers, internal commands, credentials, and unverified incident details.

## MSHLT Learning Outcomes

### 1. Programming Skills for the Workplace

Operational work strengthened skills that surround programming: reading diagnostics, querying system state, validating data relationships, distinguishing configuration from code behavior, and defining completion in observable terms. Even without an authored pull request, these are engineering skills because they require a disciplined model of how the application, task system, and stored data interact.

### 2. Fundamental NLP and Data-System Concepts

Several support requests involved analysis pipelines and calculated results. I applied the distinction between asynchronous execution, task progress, and materialized output. This distinction is important for NLP and analytics products because a request to calculate or summarize data may involve queued work, preprocessing, model or statistical execution, and persistence before a user can consume the result. A user-visible “analyzing” state is therefore a symptom, not a complete diagnosis.

### 3. Tools and Packages

The ticket evidence reflects use of Swayable’s diagnostics and setup interfaces, issue tracking in Linear, stored survey configuration, background analysis status, MongoDB-backed entities, segmentation concepts, and account and email workflows. I also relied on structured acceptance criteria and timestamps as operational tools. I do not list a package as though I modified it when no linked code change establishes that contribution.

### 4. Workplace Communication and Collaboration

Support work required concise communication between customer-facing teammates and engineering. I learned to restate impact without exposing client-sensitive details, report what had been observed rather than what I assumed, and close the loop against acceptance criteria. I also learned that “blocked from sending results” is a business-impact statement that should affect priority while remaining separate from the technical root cause.

## Challenges

The strongest repeated challenge was time-sensitive analysis state. TSUP-77 records a reanalysis that had appeared active across multiple days and an urgent need to determine completion counts. TSUP-85 records a new segment followed by a reanalysis that showed little logged progress after more than an hour. TSUP-87 similarly records slower-than-expected progress, while TSUP-92 records repeated failure after new custom segments were added. TSUP-162 records a test that remained in analysis for over an hour while results delivery was blocked. These records support a pattern of diagnosing long-running or failed analysis workflows. They do not, by themselves, establish one common root cause, so I do not claim one.

A second challenge was configuration correctness. TSUP-93 records a validation error because specified post-stratification segments lacked source data in configuration. ENG-2524 records a request to assign respondent records to predefined segments from an authorized source file. These examples required attention to identifiers and relationships. The evidence supports completion of the tickets, but it does not support publishing the underlying client data or reproducing the assignments here.

A third challenge was distinguishing result-generation issues from display or access issues. ENG-2436 records one metric missing for one treatment and an acceptance criterion of either restoring the metric or explaining why it was absent. ENG-2449 and ENG-2451 concern user account creation or login. These are materially different failure categories even when each blocks delivery. Treating them as separate layers helped avoid a generic “production issue” response.

The principal evidence limitation is the absence of authored pull requests and detailed public postmortems. Ticket completion shows that I handled assigned operational work, but it does not prove a particular internal command, data mutation, or root cause. I therefore do not invent those details.

## Outcomes

All listed records were part of completed support or engineering work during the internship period. The directly documented outcomes include completion of reanalysis-related requests, removal of a stated validation blocker, handling of user-access requests, investigation of a missing metric, and completion of requested segment assignment work. Several tickets moved from urgent or client-blocking states to “Done” on the same day, while others required a longer coordination window.

The broader outcome was increased operational fluency. I became faster at converting a report into an acceptance condition, checking the appropriate system layer, and communicating status without overclaiming. I also saw recurring areas where runbooks and observability could reduce response time. Some tickets explicitly stated that no runbook existed or had not been used. That observation does not prove that a new runbook was delivered as part of this work, but it provides a concrete direction for future process improvement.

## Professional Practice

This theme taught me that production support is not secondary to software engineering. It is where system assumptions meet time pressure, incomplete information, and customer commitments. The most important professional habit was evidence discipline. I separated what the ticket reported, what the system showed, what action was authorized, and what result was verified.

I also practiced confidentiality. The source records contain organization names, individual email addresses, internal links, and record identifiers. None is necessary to demonstrate the learning outcome, so I removed them from this entry. Sanitization is not only a portfolio requirement; it is part of responsible handling of production information.

Finally, I learned to recognize the limits of individual incident work. Completing a request can restore delivery without explaining every systemic cause. A mature follow-up may require a code change, additional monitoring, or a runbook, but I should claim those only when there is evidence that I performed them.

## Code Reference

There are no authored pull requests linked to this theme. The primary references are private Linear tickets TSUP-77, TSUP-85, TSUP-87, TSUP-92, TSUP-93, TSUP-162, ENG-2436, ENG-2449, ENG-2451, ENG-2514, and ENG-2524, available only to authorized Swayable viewers.

The pseudocode in Technical Approach documents my sanitized diagnostic method. It is intentionally not presented as deployed application code.
