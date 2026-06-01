# Medical Observational Domain Demo

This demo topic exercises the `medical_observational` domain profile without
using real patient data:

```text
Retrospective cohort study of risk factors for mortality in trauma registry patients
```

The detector routes this topic to `medical_observational` because it contains
domain-specific phrases such as `retrospective cohort` and `trauma registry`.

## Intended Behavior

The profile and adapter add safeguards for observational clinical research:

- require an ethics/IRB or waiver status before analysis;
- require de-identification and privacy confirmation before handling real data;
- document data access, eligibility criteria, exposure, outcome, confounders,
  effect modifiers, and follow-up window before modelling;
- plan missing-data handling before model fitting;
- produce STROBE-aligned outputs such as participant flow counts, Table 1,
  effect estimates with confidence intervals, limitations, and sensitivity
  analyses.

## Synthetic Schema Only

Use synthetic fields for examples and tests:

| Field | Type | Example values |
| --- | --- | --- |
| `age_years` | numeric | 18-95 |
| `sex` | categorical | `female`, `male`, `unknown` |
| `injury_severity_score` | numeric | 1-75 |
| `shock_index` | numeric | 0.4-2.5 |
| `mechanism` | categorical | `fall`, `traffic`, `penetrating`, `other` |
| `massive_transfusion` | binary | 0, 1 |
| `mortality_30d` | binary outcome | 0, 1 |

Do not include real patient records, identifiable dates, hospital names,
medical-record numbers, free-text notes, IRB identifiers, or institution-specific
workflows in demos.

## Reference Implementation Boundary

This domain profile was informed by the public
`TUANZIDING/medical-paper-pipeline` project, but it intentionally contributes
only a lightweight AutoResearchClaw domain pack: profile metadata, prompt
adapter safeguards, detector keywords, and this demo note. It does not import
the external pipeline or replace AutoResearchClaw's 23-stage runner.
