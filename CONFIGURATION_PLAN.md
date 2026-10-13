# Configuration-driven LUMINA: plan and delivery record

Base: `f9036f7085ac53ec2739b0cf94a945beea1e3a43` (`Voss-zeji/LUMINA` main).
Branch: `feat/config-driven-workflow`.

## Approved behavior

Edit one study configuration directory, run one command, review a small trial,
explicitly confirm, then continue the batch. Keep durable recovery, request
receipts, budgets, and the existing scientific acceptance rules internally.
Use Python 3.12 TOML loading and existing dependencies. All study prompts and
question rules live outside Python. New text/numeric topics require configuration
only. Real paper API runs are excluded from this implementation.

## Goals and evidence

| Goal | Acceptance | Status / evidence |
|---|---|---|
| G1 Baseline | Existing offline suite and exact built-in prompt baseline | Passed: 386 original tests; all 10 original prompt constants retain SHA-256 |
| G2 Configuration | TOML, paper lists, secrets separation, validation, examples | Implemented; loader validation and Chinese-path checks pass |
| G3 Scientific integration | Custom questions propagate through every stage, rules and fingerprints | Passed custom topic with numeric Q1/text Q2; independent evaluator reads frozen rules |
| G4 Simple execution | Check, trial, confirmation, resume, completed-run reuse | Passed in-process and real child-process TOML workflows without network |
| G5 Documentation and compatibility | Bilingual quickstart, legacy entry retained, migration explained | Updated READMEs and CONFIGURATION.md; links, fences and TOML parsing checked |
| G6 Acceptance | New topic, prior domain parity, isolation, budgets, no secret leaks, Windows/Linux | Local 398-test regression and final targeted checks passed; live cross-platform evidence: [PR #6 checks](https://github.com/Voss-zeji/LUMINA/pull/6/checks) |

## Contracts

- Each study contains `project.toml`, `papers.txt`, `questions.toml`, and a Git-ignored
  `secrets.local.toml` (a keyless example is supplied).
- Configuration is loaded centrally. Research definitions and inputs are frozen
  per run; resume rejects incompatible changes. Credentials are never frozen.
- Question definitions select existing `text` or `numeric` behavior and configure
  item names, allowed values, units, and experiment requirements. No new scientific
  algorithms or arbitrary nested response format are introduced.
- Preserve Tvfy AND Tbsl, distinct generating-model votes, unaccepted ties, and
  separate units/experiments. Reuse existing trial results in the batch.
- Legacy Python configuration/CLI stays compatible; old run artifacts are not
  rewritten. Incompatible old runs require the original revision or a new run.
- Noninteractive execution stops at confirmation. Unknown paid requests never
  receive blind retries or automatic approval.

## Implementation order

1. Capture baseline; build config/prompt and question-rule modules in parallel.
2. Wire pipeline, runtime contracts, fingerprints, and child-process loading.
3. Add the simple study entry and its offline end-to-end tests.
4. Update README examples and migration instructions.
5. Run acceptance checks and independent review, fix findings, record evidence.

## Verification (2026-10-08)

- Python 3.12.13 in the isolated worktree `.venv`; repository requirements unchanged.
- Original revision: 386 tests passed (261.071 s), network disabled.
- Integrated regression: 398 tests passed (172.758 s), network disabled.
- Review regression reproduced an eager default that read built-in Aqua prompts
  even for custom studies; corrected to resolve only the requested templates.
  Added the independent-config regression; 14 focused workflow/config/controller
  checks passed after the fix (41.776 s).
- Added portable domain-name validation because domain names form output filenames;
  12 contract/config checks passed (0.227 s).
- Ruff E4/E7/E9/F, patch whitespace, README links/fences, and TOML parser checks pass.
- Real child-process test uses only synthetic responses with outbound sockets
  blocked. It covers Chinese paths, TOML loading, trial pause, approval, completion,
  and an unchanged repeat without additional requests.
- Independent review checked configuration boundaries and the scientific call chain.
  No real-paper API workload has been launched.
- Final review confirmed the scientific dispatch and provenance wiring. Whole-task
  rejection for mixed invalid responses is intentional and tested; original Wildfire
  validation remains unchanged rather than introducing a new scientific requirement.
- Initial GitHub run `37756701192`: Ubuntu passed all 399 tests; Windows passed 398
  with one test-only mismatch between an 8.3 temporary path and its resolved long
  name. The path assertion now normalizes both sides; no production change was needed.
- Delivery: [PR #6](https://github.com/Voss-zeji/LUMINA/pull/6), branch
  `feat/config-driven-workflow`. Its Windows and Ubuntu checks are the live
  cross-platform acceptance record for the latest revision; both must be green.
  PR #6 was merged into main at `c32c7319f820a20fe464cd4c164bf7aa86b22639`.

## PDF-first installation follow-up (user approved)

Default inputs are PDFs, with Markdown generated by Marker and an explicitly
recorded pypdf fallback. Clone-and-install supports a custom Python 3.12 venv.
This follow-up remains in this same plan/acceptance record.

| Goal | Acceptance | Evidence |
|---|---|---|
| P1 Merge configuration work | PR #6 merged into Voss main | `c32c731`, confirmed on GitHub |
| P2 Custom venv installation | Create/reuse target venv; install/probe default Marker and core; honest fallback | Passed: fresh custom venv installed default profile, pip check and actual Marker API imports |
| P3 PDF preparation | Source PDF to derived Markdown, converter provenance, no silent empty pages | 20 preparation/PDF tests passed, including real pypdf conversion |
| P4 Delivery | Bilingual instructions, cross-platform offline tests, merge | Local 431 tests and Ruff passed; live cross-platform/merge evidence: [PR #7](https://github.com/Voss-zeji/LUMINA/pull/7) |

Default PDF requirement is pinned to Marker 2.0.0 after inspecting its published
wheel API. The dependency set resolves for Python 3.12 on this Windows host.
pypdf is a text-only fallback; no new OCR quality or scientific accuracy claim is made.

Fresh default installation evidence: Python 3.12.13, Marker 2.0.0, pypdf 6.19.0,
OpenAI SDK 2.54.0 and Torch 2.14.1. All 431 tests passed in that environment
(188.059 s), with network blocked for the tests. The setup smoke installed packages
only; Marker OCR weights were not downloaded and real scientific APIs were not called.
The existing local pip mirror stalled; only the smoke process used official PyPI
via `PIP_INDEX_URL`, without modifying global pip settings.

The independent review confirmed UID-based PDF provenance, fallback routing,
empty-page rejection, configuration fingerprints, and isolated pip targeting.
Its remaining first-use model-download timing note is now in both READMEs.

## Interactive terminal follow-up (approved 2026-10-13)

Base main: `5a5e591`. Branch: `feat/interactive-cli`.

One Python 3.12 launcher prepares/reuses the selected venv, shows an ASCII welcome,
and guides study setup in Chinese/English (Chinese default). New/edit/resume are
separate actions. Stage settings are completed before execution and saved in the
existing four study files. PDF directory selection, exclusions, prompt preview and
editor, displayed defaults, back navigation and draft save are required.

Budgets may be omitted independently (no corresponding total cap); prices may be
unknown, never zero by implication. Default verification threshold is 1 and consensus
is a strict model majority for new studies; existing values remain unchanged.
Requests retain finite timeout/output bounds, receipted accounting, a three-attempt
ceiling, self-exclusion and explicit trial-to-batch approval. No native-PDF API path.

| Goal | Acceptance | Evidence |
|---|---|---|
| I1 Launcher and wizard | Reuse install.py; healthy environments do not reinstall; bilingual grouped setup, defaults, draft/edit/resume | Passed focused tests and actual Windows terminal demonstration; Marker/fallback profile verified |
| I2 Optional budgets and quotes | None-safe caps/costs, model request bounds independent of prices, finite old caps preserved | Passed optional/legacy budget, quote, and two-model/two-paper/two-round trial/batch/resume tests |
| I3 Runtime feedback | Honest phase/round counts, waits, cached requests, round artifacts, no JSON protocol or deadline regression | Passed read-only progress, slow observer, large child receipt, repeated real runtime wait and round artifact tests |
| I4 Full logging | Redacted setup/input provenance/request/response/use/error/output records; linked startup log | Passed nested secret redaction, private draft and setup-log linking tests |
| I5 Delivery | Offline Windows/Ubuntu checks, terminal demonstration, bilingual documentation and review | Passed: 485 tests and Ruff on each platform; [CI 38328836225](https://github.com/Voss-zeji/LUMINA/actions/runs/38328836225), code revision `d993785` |

Generic quotes are public JSON URLs or JSON files with a `quotes` array. Each row
contains `base_url`, `model`, `currency`, `input_per_million`, `output_per_million`,
and `as_of`. Only exact endpoint/model matches are eligible; imports require preview
and confirmation. Capture source/fetch time; do not convert currencies or infer proxy
prices from an upstream vendor. Missing/unrecognized prices can remain unknown. A
money cap requires complete, comparable pricing.

New model fields `max_input_tokens` (optional) and `max_output_tokens` (default 4096,
embedding default 0) define request limits independently of prices. Explicit legacy
pricing bounds remain authoritative for legacy configurations. Unset input limits use
a conservative byte-based request reservation; unset total limits are represented as
null in the resolved snapshot, never Infinity. New study files may omit budget/rate
fields; legacy full configurations remain accepted.

Implementation ownership: parent owns launcher, wizard, quote import, install/deps,
documentation and integration; budget worker owns budget/contracts/runtime/config
and its tests; progress worker owns controller/worker/report/progress and its tests.
Workers coordinate shared interfaces rather than editing each other's files.


Current integration evidence (2026-10-13):
- 47 focused UI/install/quote tests passed, including silent-install heartbeat,
  Ctrl-C child cleanup, hidden keys, private draft exit, TOML round-trip and quote rejection.
- Actual Windows PTY used `start.py --venv .venv --lang en --config <synthetic-study>`:
  healthy venv skipped pip installation; ASCII welcome and grouped study defaults
  appeared; draft exit saved to `runs/_setup/`. A second launch without `--lang`
  remembered English. No real API call or OCR weight download occurred.
- Added end-to-end coverage for omitted caps/prices and retained legacy token bounds.
- Local full regression passed. GitHub Ubuntu/Windows results are recorded in the final acceptance below.

Independent review fixes: structured auth-field redaction, Marker-aware startup health
with persistent explicit fallback, malformed pricing rejection, and real wait-event
producer/renderer pairing. Legacy quotes may retain their common implicit monetary
unit for compatibility; explicit mixed currencies are rejected, and imports require
currency plus endpoint binding. Known rates still require a price basis.

A second actual Windows PTY demonstration used the custom `pdf-install-smoke` venv
with Marker installed and Chinese UI. Imports passed, installation was skipped,
ASCII/group defaults appeared and draft exit succeeded. No OCR weights or real
research requests were started.

Post-review display correction: phase/resume stage detection now uses event order,
and cache counts are bounded by the current stage/round invocation. A same-scope
trial-to-batch/resume regression verifies that old cache events cannot inflate the
new display. The 22 progress/controller/configuration tests passed (67.268 s).
Delivery PR: https://github.com/Voss-zeji/LUMINA/pull/8 (cross-platform checks passed).

Final terminal execution evidence: a network-blocked mock run in a short Windows
output directory completed the six stages, paused for explicit trial approval,
then completed the batch with zero new requests (11 total mock requests, 145 tokens,
2 accepted rows). Unknown prices remained a null total. A first attempt under the
deep checkout encountered Windows' path-length limit; the bilingual READMEs now
recommend a short results root when long-path support is disabled. Display paths
are condensed to output directories; previews omit internal candidate IDs, with
complete data retained in result files. Seven final progress tests and Ruff passed.

Final Ubuntu/Windows acceptance evidence is maintained in PR #8's Checks and
validation section. Merge is contingent on both platform checks passing.

Unchanged-edit regression: pressing Enter throughout an existing study must preserve
its exact frozen fingerprint, including integer/float representations. This test
first reproduced a drift from numeric coercion, then passed after preserving original
numeric values and untouched rates. All 25 wizard/quote/optional-budget tests passed.

Cross-platform issue found by CI run 38327688949: Ubuntu passed all 484 tests;
English Windows ran all 484 and reported six launcher UnicodeEncodeError failures
because redirected stdout used cp1252. The product now configures UTF-8 output at
the launcher, wizard and study boundaries, and its probes explicitly use UTF-8.
A subprocess regression reproduces cp1252 without changing CI locale. All 37
launcher/wizard/configuration/study tests passed under forced cp1252 (64.054 s).
Final cross-platform rerun is linked from PR #8.

## Final interactive CLI acceptance

All I1-I5 acceptance goals passed. Verified code: `d993785d22bd5ab128c390cc87d40f94a97e7e8e`.
[GitHub Actions run 38328836225](https://github.com/Voss-zeji/LUMINA/actions/runs/38328836225)
completed successfully on both platforms:

- Ubuntu: 485 tests passed in 155.828 s; Ruff passed.
- Windows: 485 tests passed in 493.948 s; Ruff passed.
- Local Windows with forced cp1252: 485 tests passed in 224.005 s.
- Actual terminal: custom Marker venv, bilingual welcome/defaults, language memory,
  draft exit, and offline trial/approval/batch/resume demonstrated.
- No real scientific API calls or OCR model-weight downloads were performed.

The final acceptance-record commit changes only this Markdown file; the delivered
program and tests are byte-for-byte identical to the cross-platform-tested revision.
Review and merge record: [PR #8](https://github.com/Voss-zeji/LUMINA/pull/8).
