# Evaluation datasets

Question sets for `answerability_eval.py`. Each file lists questions with an
expected answerable/unanswerable label and, optionally, the pages that hold the
evidence. Page accuracy is reported separately from answerability accuracy.

    python answerability_eval.py "/path/to/paper.pdf" --eval evaluations/real_papers/<file>.json

## `real_papers/` — regression datasets for actual PDFs

Run against the real paper with the real models. These are the sets the
release was validated on.

| File | Paper | Questions |
| --- | --- | --- |
| `mobile_learning.json` | *A look at advanced learners' use of mobile devices for English language study* (Kruk, 2017) | 12 |
| `publication_language.json` | *The language of (future) scientific communication* (van Weijen, 2012) | 10 |

The PDFs themselves are not committed. Supply your own copy.

## `synthetic_fixtures/` — datasets for the built-in test fixtures

These describe **synthetic papers defined in `fixtures.py`**, not real
publications. They exist so the pipeline can be regression-tested without a PDF
or a model, and they are wired into the automated tests.

| File | Fixture | Used by |
| --- | --- | --- |
| `mobile_devices.json` | `MOBILE_PAPER_PAGES` | `test_answerability.py` |
| `interview_study.json` | `INTERVIEW_PAPER_PAGES` | manual runs only |

Do not read results from these as evidence about real-paper behaviour.
