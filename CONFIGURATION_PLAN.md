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
| P4 Delivery | Bilingual instructions, cross-platform offline tests, merge | Local 431 tests and Ruff passed; cross-platform/merge evidence follows the PDF-first PR |

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
