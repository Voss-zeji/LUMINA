from __future__ import annotations

from pathlib import Path
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
from langchain_text_splitters import MarkdownTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

from . import prompts
from .common import (atomic_output, candidate_id, error_details, fingerprint, filename_token,
                     invalid_path, load_json, paper_markdowns, read_composite,
                     refineJsonString, save_json, turnIntoPureText, write_dataframe, write_invalid_text)
from .llm import embedding_response, llm_requery
from .utils import model_name, sleep_for_rate_limit

_INVALID_EVIDENCE = {'nan', 'na', 'n/a', 'none', 'null', ''}


def save_embeddings(file_path: str, array: np.ndarray) -> None:
    with atomic_output(file_path) as temp:
        with temp.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)


def load_embeddings(file_path: str):
    try:
        return np.load(file_path, allow_pickle=False)
    except (EOFError, OSError, ValueError):
        return None


def embedding_file(domain_cfg: dict, paper_prefix: str, chunk_size: int, overlap_percent: int) -> str:
    return str(Path(domain_cfg['embedding_dir']) / f'Paper{paper_prefix}' /
               f'ChunkSize{chunk_size:05d}_Overlap{overlap_percent:03d}.npy')


def _splitter(run_cfg: dict) -> MarkdownTextSplitter:
    size, overlap = run_cfg['chunk_size'], run_cfg['overlap_percent']
    if type(size) is not int or size <= 0 or not 0 <= overlap < 100:
        raise ValueError('chunk_size must be a positive integer and overlap_percent in [0,100)')
    return MarkdownTextSplitter(chunk_size=size, chunk_overlap=int(size * overlap / 100))


def _chunks_for_paper(markdown_path: str, run_cfg: dict) -> list[str]:
    return _splitter(run_cfg).split_text(turnIntoPureText(markdown_path))


def _embedding_spec(embedding_model: dict, llm_settings: dict, run_cfg: dict) -> dict:
    provider = llm_settings.get(embedding_model.get('source'), {})
    return dict(version=2, model=embedding_model.get('model'), source=embedding_model.get('source'), endpoint=embedding_model.get('url') or provider.get('url'),
                chunk_size=run_cfg['chunk_size'], overlap_percent=run_cfg['overlap_percent'])


def _valid_vectors(array, count: int) -> bool:
    return (isinstance(array, np.ndarray) and array.ndim == 2 and array.shape[0] == count
            and array.shape[1] > 0 and np.issubdtype(array.dtype, np.number)
            and np.isfinite(array).all() and (np.linalg.norm(array, axis=1) > 0).all())


def _paper_embeddings(domain_cfg: dict, paper: str, chunks: list[str], embedding_model: dict,
                      llm_settings: dict, run_cfg: dict, runtime=None) -> np.ndarray:
    if not chunks:
        raise ValueError(f'empty chunks for Paper {paper}')
    output = Path(embedding_file(domain_cfg, paper, run_cfg['chunk_size'], run_cfg['overlap_percent']))
    meta_file = output.with_suffix('.meta.json')
    key = None if runtime is None else runtime.key('embeddings', paper,
                                                    source_model=embedding_model['model'],
                                                    round_index=run_cfg['round_index'])
    signature = fingerprint(dict(spec=_embedding_spec(embedding_model, llm_settings, run_cfg), chunks=chunks))
    metadata = load_json(meta_file) if meta_file.exists() else None
    cached = load_embeddings(str(output))
    if metadata and metadata.get('fingerprint') == signature and _valid_vectors(cached, len(chunks)):
        if metadata.get('dimensions') == cached.shape[1]:
            # Reusable only when the science matches AND the ledger still vouches for the
            # bytes; otherwise rebuild from cached per-chunk receipts without new API calls.
            if runtime is None or runtime.valid(key):
                if runtime is not None:
                    runtime.completed(key, [str(output), str(meta_file)], cached=True)
                return cached
            print(f'Paper {paper}: embedding matrix changed, rebuilding from cached chunk receipts')
    scope = nullcontext() if runtime is None else runtime.task_scope(key)
    transport = {} if runtime is None else {"runtime": runtime}
    with scope:
        vectors = np.array([embedding_response(chunk, embedding_model, llm_settings, **transport)
                            for chunk in chunks], dtype=float)
        if not _valid_vectors(vectors, len(chunks)):
            raise ValueError(f'invalid embedding shape/numbers for Paper {paper}')
        save_embeddings(str(output), vectors)
        save_json(meta_file, dict(fingerprint=signature, dimensions=vectors.shape[1], chunks=len(chunks)))
        if runtime is not None:
            runtime.completed(key, [str(output), str(meta_file)])
    return vectors


def generate_embeddings_for_domain(domain_cfg: dict, embedding_model: dict, llm_settings: dict,
                                    run_cfg: dict, runtime=None) -> None:
    transport = {} if runtime is None else {"runtime": runtime}
    for paper, path in paper_markdowns(domain_cfg['markdown_dir']).items():
        chunks = _chunks_for_paper(str(path), run_cfg)
        _paper_embeddings(domain_cfg, paper, chunks, embedding_model, llm_settings, run_cfg, **transport)
        print(f'embedded Paper {paper} ({len(chunks)} chunks)')


def independent_verifiers(llm_dicts: dict, selected_model_names: list[str], input_model: str) -> list[dict]:
    return [llm for llm in llm_dicts.values()
            if model_name(llm) in selected_model_names and model_name(llm) != input_model]


def verifier_signatures(llm_dicts: dict, llm_settings: dict, embedding_model: dict, run_cfg: dict) -> dict[str, str]:
    result = {}
    for llm in llm_dicts.values():
        provider = llm_settings.get(llm.get('source'), {})
        result[model_name(llm)] = fingerprint(dict(version=2, model=llm['model'], source=llm.get('source'), endpoint=provider.get('url'),
            json_mode=provider.get('supports_json_mode', True), temperature=run_cfg['temperature'],
            embedding=_embedding_spec(embedding_model, llm_settings, run_cfg), extension=run_cfg['text_extension'],
            system=prompts.message_system_ragQuery, checker=prompts.checker_requery))
    return result


def _successful_cross_output(output_file: Path, expected: str | None = None) -> bool:
    try:
        result = pd.read_csv(output_file, sep='\t', keep_default_na=False, dtype={'direct_quote': str})
        if len(result) != 1 or not {'existing_flag', 'direct_quote', 'verification_fingerprint'}.issubset(result):
            return False
        flags = pd.to_numeric(result['existing_flag'], errors='coerce')
        quote_ok = ((flags == 0) | result.direct_quote.str.strip().ne('')).all()
        return flags.isin([0, 1]).all() and quote_ok and (expected is None or result.verification_fingerprint.eq(expected).all())
    except (OSError, ValueError, pd.errors.ParserError):
        return False


def cross_validate_domain(domain: str, domain_cfg: dict, llm_dicts: dict, llm_settings: dict,
                          embedding_model: dict, run_cfg: dict, selected_model_names: list[str],
                          runtime=None) -> None:
    papers = paper_markdowns(domain_cfg['markdown_dir'])
    signatures = verifier_signatures(llm_dicts, llm_settings, embedding_model, run_cfg)
    # Composite rows carry the short display name; task identity needs the full model ID.
    full_ids = {model_name(llm): llm['model'] for llm in llm_dicts.values()}
    transport = {} if runtime is None else {'runtime': runtime}
    prepared = {}
    failed = []
    for question_index in range(1, len(prompts.questions_for_domain(domain)) + 1):
        file = Path(domain_cfg['composite_dir']) / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
        if not file.exists():
            failed.append(f'missing composite: {file}')
            continue
        frame = read_composite(file)
        for all_index, row in frame.iterrows():
            input_model, paper = row['model'], row['paper_index']
            if input_model not in selected_model_names:
                continue
            evidence = str(row.get('evidence', ''))
            if evidence.strip().lower() in _INVALID_EVIDENCE:
                continue
            if paper not in papers or row['paper_fingerprint'] != fingerprint(turnIntoPureText(papers[paper])):
                failed.append(f'Paper {paper}: changed/missing source; rerun examiner/composite')
                continue
            cid = candidate_id(row)
            pending = []
            for llm in independent_verifiers(llm_dicts, selected_model_names, input_model):
                output_model = model_name(llm)
                verification = fingerprint(dict(candidate=cid, verifier=signatures[output_model]))
                output = (Path(domain_cfg['crosser_dir']) / f'Paper_{paper}' / f'Q{question_index:02d}' /
                          f"Candidate_{cid}==Output_{filename_token(output_model)}.csv")
                verifier_key = None if runtime is None else runtime.key(
                    'cross', paper, question_index, source_model=full_ids.get(input_model, input_model),
                    verifier_model=llm['model'], round_index=run_cfg['round_index'], candidate=cid)
                if _successful_cross_output(output, verification):
                    # A sound CSV whose bytes still match the ledger needs no new paid vote.
                    if runtime is None or runtime.valid(verifier_key):
                        if runtime is not None:
                            runtime.completed(verifier_key, [str(output)], cached=True)
                        continue
                pending.append((llm, output_model, output, verification, verifier_key))
            if not pending:
                continue  # Do not pay for evidence embeddings on an unchanged resume.
            if paper not in prepared:
                chunks = _chunks_for_paper(str(papers[paper]), run_cfg)
                vectors = _paper_embeddings(domain_cfg, paper, chunks, embedding_model, llm_settings,
                                            run_cfg, **transport)
                prepared[paper] = (chunks, vectors)
            chunks, vectors = prepared[paper]
            evidence_key = None if runtime is None else runtime.key(
                'evidence_embedding', paper, question_index,
                source_model=full_ids.get(input_model, input_model),
                round_index=run_cfg['round_index'], candidate=cid)
            with nullcontext() if runtime is None else runtime.task_scope(evidence_key):
                description = np.asarray(
                    embedding_response(evidence, embedding_model, llm_settings, **transport), dtype=float)
            if description.ndim != 1 or len(description) != vectors.shape[1] or not np.isfinite(description).all() or np.linalg.norm(description) == 0:
                raise ValueError(f'Paper {paper}: evidence vector incompatible with cached embedding model')
            similarities = cosine_similarity(description.reshape(1, -1), vectors)[0]
            chunk_id = int(np.argmax(similarities))
            extension = run_cfg['text_extension']
            if type(extension) is not int or extension < 0:
                raise ValueError('text_extension must be a nonnegative integer')
            context = '\n\n'.join(chunks[max(0, chunk_id - extension):chunk_id + extension + 1])
            for llm, output_model, output, verification, verifier_key in pending:
                with nullcontext() if runtime is None else runtime.task_scope(verifier_key):
                    start, raw = time.time(), None
                    try:
                        query_kwargs = {'temperature': run_cfg['temperature'], **transport}
                        raw, token = llm_requery(llm, llm_settings, prompts.message_system_ragQuery.strip(),
                            prompts.checker_requery.format(answer=evidence, context=context, key_topic=row['item']).strip(),
                            **query_kwargs)
                        result = refineJsonString(raw)
                        if result.get('existing_flag') not in (0, 1) or not isinstance(result.get('direct_quote'), (str, type(None))):
                            raise ValueError('verifier requires existing_flag 0/1 and scalar direct_quote')
                        if result['existing_flag'] == 1 and not (result['direct_quote'] or '').strip():
                            raise ValueError('positive verifier vote requires a nonempty direct_quote')
                        # Keep existence checking, not a new numeric/unit fact-checking task.
                        result = dict(existing_flag=int(result['existing_flag']), direct_quote=result['direct_quote'])
                    except Exception as exc:
                        if runtime is not None:
                            runtime.reraise_control(exc)  # never let a control stop become a -1 vote
                        detail = error_details(exc, llm_settings)
                        failed.append(f'Paper {paper} Q{question_index} {input_model}->{output_model}: {detail}')
                        if runtime is not None:
                            # Explicitly failed, not a -1 vote: an API error is not evidence.
                            runtime.failed(verifier_key, detail)
                        write_invalid_text(output, f'ERROR {detail}\nRAW: {raw}')
                        result, token = dict(existing_flag=-1, direct_quote=None), None
                    else:
                        invalid_path(output).unlink(missing_ok=True)
                    result.update(paper_index=paper, question_index=question_index, item_raw_index=all_index,
                              candidate_id=cid, input_model=input_model, output_model=output_model,
                              verification_fingerprint=verification, evaluate_token=token,
                              time_consumption=time.time() - start, max_similarities=float(similarities[chunk_id]))
                    write_dataframe(pd.DataFrame([result]), output)
                    # Only a real 0/1 vote is a finished task; a -1 API failure is not.
                    if runtime is not None and int(result['existing_flag']) in (0, 1):
                        runtime.completed(verifier_key, [str(output)])
                    sleep_for_rate_limit(llm, low_limit_seconds=5, high_limit_seconds=.1, **transport)
    if failed:
        raise RuntimeError('cross failed: ' + '; '.join(failed))


def aggregate_cross_scores(domain_cfg: dict, selected_model_names: list[str],
                           expected_verifiers: dict[str, str] | None = None) -> pd.DataFrame:
    votes = {}
    for file in sorted(Path(domain_cfg['crosser_dir']).glob('Paper_*/Q*/Candidate_*==Output_*.csv')):
        try:
            part = pd.read_csv(file, sep='\t', keep_default_na=False, dtype={'paper_index': str, 'direct_quote': str})
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f'unreadable verifier output {file}: {exc}') from exc
        required = {'candidate_id', 'input_model', 'output_model', 'existing_flag', 'direct_quote', 'verification_fingerprint'}
        if not required.issubset(part):
            continue  # Legacy row-index-only votes cannot establish current evidence support.
        part = part[part.input_model.isin(selected_model_names) & part.output_model.isin(selected_model_names)
                    & (part.input_model != part.output_model)]
        part = part[pd.to_numeric(part.existing_flag, errors='coerce').isin([0, 1])]
        part = part[(pd.to_numeric(part.existing_flag) == 0) | part.direct_quote.str.strip().ne('')]
        for vote in part.to_dict('records'):
            votes.setdefault((vote['candidate_id'], vote['input_model']), []).append(vote)
    summaries = []
    for file in sorted(Path(domain_cfg['composite_dir']).glob(f"{domain_cfg['composite_prefix']}_Q*.xlsx")):
        frame = read_composite(file)
        frame = frame.drop(columns=[c for c in frame if c.startswith('flag_') or c.startswith('cross_score') or c == 'item_raw_index'])
        for name in selected_model_names:
            frame[f'flag_{name}'] = np.nan
        frame['cross_score'] = np.nan
        for index, row in frame.iterrows():
            cid = candidate_id(row)
            frame.at[index, 'candidate_id'] = cid
            matching = votes.get((cid, row['model']), [])
            if matching and expected_verifiers is None:
                raise ValueError('current verifier signatures required; run cross/ensemble through CLI')
            flags = {}
            for vote in matching:
                expected = fingerprint(dict(candidate=cid, verifier=expected_verifiers.get(vote['output_model'])))
                if vote['verification_fingerprint'] != expected:
                    continue
                flags[vote['output_model']] = max(flags.get(vote['output_model'], 0), int(vote['existing_flag']))
            if flags:
                for name, flag in flags.items():
                    frame.at[index, f'flag_{name}'] = flag
                frame.at[index, 'cross_score'] = sum(flags.values())
        write_dataframe(frame, file)
        summaries.append(frame.loc[frame['cross_score'].notna(), ['paper_index', 'question_index', 'candidate_id', 'model', 'cross_score']
                                   + [f'flag_{name}' for name in selected_model_names]])
    result = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    write_dataframe(result, Path(domain_cfg['crosser_dir']) / 'cross_scores.xlsx')
    return result
