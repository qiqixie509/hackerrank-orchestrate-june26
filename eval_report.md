# Evaluation Report

Rows scored: **20**

## claim_status (primary)

- accuracy: **1.000**
- macro F1: **1.000**

| class | precision | recall | f1 |
|---|---|---|---|
| contradicted | 1.00 | 1.00 | 1.00 |
| not_enough_information | 1.00 | 1.00 | 1.00 |
| supported | 1.00 | 1.00 | 1.00 |

**Confusion matrix:**

| gold \ pred | contradicted | not_enough_information | supported |
|---|---|---|---|
| **contradicted** | 5 | 0 | 0 |
| **not_enough_information** | 0 | 2 | 0 |
| **supported** | 0 | 0 | 13 |

## Categorical accuracy

| field | accuracy |
|---|---|
| claim_status | 1.000 |
| issue_type | 1.000 |
| object_part | 1.000 |
| evidence_standard_met | 1.000 |
| valid_image | 1.000 |

## Severity (ordinal)

| metric | value |
|---|---|
| exact | 1.000 |
| within_one | 1.000 |
| mae_ordinal | 0.000 |

## Set fields

| field | micro_P | micro_R | micro_F1 | jaccard | exact_set |
|---|---|---|---|---|---|
| risk_flags | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| supporting_image_ids | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Mismatched rows (claim_status)

None - all claim_status predictions matched.
