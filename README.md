# LUMINA

We piloted LUMINA in two distinct tasks—water (aquaculture) and fire—to construct a LLM-based emission factor database. All extraction tasks follow standard meta-analysis protocols.

## What is included

- `lumina/preparation.py`: PDF -> markdown conversion hook (marker), markdown truncation before References/Acknowledgments/Appendix, token audit.
- `lumina/common.py`: shared utilities (JSON salvage, file naming, paper-prefix parsing, invalid-result placeholders).
- `lumina/prompts.py`: all aqua + wildfire examiner prompts and both cross-validation checker prompts.
- `lumina/examiner.py`: Stage A examiner, per-model structured JSON extraction for aqua and wildfire.
- `lumina/composite.py`: baseline composite builder, including `_invalid.txt` salvage.
- `lumina/cross_validation.py`: Stage B cross-validation with embedding retrieval (chunk markdown, cache `.npy` embeddings, embed each evidence, retrieve nearest chunks with `text_extension`, verify with other models).
- `lumina/ensemble.py` + `lumina/ensemble_utils_meta.py` + `lumina/ensemble_utils_value.py`: Stage 3 consensus/ensemble for meta items and numeric items.
- `run_pipeline.py`: CLI chaining the stages.

All API keys, base URLs, endpoints, and local data paths are placeholders. Copy `config.example.py` to `config.py` and fill it locally; never commit real secrets.

## Expected local data layout

```text
data/
  pdfs/
    aqua/*.pdf
    wildfire/*.pdf
  mds/
    aqua/*.md
    wildfire/*.md
output/
  examiner/{aqua,wildfire}/Paper_XX/*.csv
  composite/{aqua,wildfire}/*_CrossValidation_Q*.xlsx
  embeddings/{aqua,wildfire}/PaperXX/ChunkSize02048_Overlap020.npy
  crosser/{aqua,wildfire}/Paper_XX/QXX/*.csv
  ensemble/{aqua,wildfire}/*.xlsx
```

Paper prefix rule: the numeric/ID head before the first `_` or `-`; numeric heads are zero-padded to two digits.

## Run stages

```bash
python run_pipeline.py --domain wildfire --stage prepare
python run_pipeline.py --domain wildfire --stage examiner
python run_pipeline.py --domain wildfire --stage composite
python run_pipeline.py --domain wildfire --stage embeddings
python run_pipeline.py --domain wildfire --stage cross
python run_pipeline.py --domain wildfire --stage ensemble
```

Use `--domain aqua` for the aqua pipeline. `--stage all` runs the full chain in the order above.

## Validation without secrets

The package is import-safe with blank config. A real run requires local PDFs/markdowns and valid OpenAI-compatible chat + embedding endpoints in `config.py`.
