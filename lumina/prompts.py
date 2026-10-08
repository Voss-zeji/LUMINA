"""Load study prompts and question rules from data files, without global overrides."""
from __future__ import annotations

import copy
import string
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / 'configs'
_VARIABLES = {'system': {'domain'}, 'instruction': {'content'}, 'verifier_system': set(),
              'verifier_instruction': {'context', 'answer', 'key_topic'},
              'verifier_full': {'context', 'answer'}}
_FIELDS = {'id', 'prompt', 'kind', 'items', 'allowed_values', 'require_unit',
           'require_experimental', 'emission_type'}


def _builtin(domain: str) -> dict:
    domain = domain.lower()
    name = 'aqua' if domain.startswith('aqua') else 'wildfire' if domain.startswith('wild') else None
    if name is None:
        raise ValueError(f'Unsupported domain: {domain}; configure a question_set')
    with (_ROOT / name / 'questions.toml').open('rb') as stream:
        return tomllib.load(stream)


def validate_definition(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {'templates', 'questions', 'legacy'}:
        raise ValueError('Question file requires templates and questions; unknown fields are not supported')
    templates = value.get('templates')
    if not isinstance(templates, dict) or set(templates) != set(_VARIABLES):
        raise ValueError('templates must contain system, instruction, verifier_system, verifier_instruction, verifier_full')
    for name, variables in _VARIABLES.items():
        text = templates[name]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'template {name} must be nonempty text')
        if name == 'verifier_system':
            continue
        try:
            fields = {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}
            if fields != variables:
                raise ValueError(f'template {name} requires exactly placeholders {sorted(variables)}')
            text.format(**dict.fromkeys(variables, 'test'))
        except (KeyError, IndexError, AttributeError, ValueError):
            raise ValueError(f'invalid placeholders/braces in template {name}') from None
    questions = value.get('questions')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions must be a nonempty list')
    normalized, ids = [], set()
    for question in questions:
        if not isinstance(question, dict) or set(question) - _FIELDS:
            raise ValueError('unknown question fields')
        rule = dict(items=[], allowed_values=[], require_unit=False, require_experimental=False) | question
        for field in ('id', 'prompt'):
            if not isinstance(rule.get(field), str) or not rule[field].strip():
                raise ValueError(f'question {field} must be nonempty text')
        if rule['id'] != rule['id'].strip() or rule['id'] in ids:
            raise ValueError('question id must be unique and have no surrounding whitespace')
        ids.add(rule['id'])
        if rule.get('kind') not in {'text', 'numeric'}:
            raise ValueError('question kind must be text or numeric')
        for field in ('items', 'allowed_values'):
            entries = rule[field]
            if not isinstance(entries, list) or any(not isinstance(v, str) or not v.strip() for v in entries):
                raise ValueError(f'{field} must be a list of nonempty strings')
            if len(set(entries)) != len(entries):
                raise ValueError(f'{field} must not contain duplicates')
        for field in ('require_unit', 'require_experimental'):
            if type(rule[field]) is not bool:
                raise ValueError(f'{field} must be boolean')
        if 'emission_type' in rule and (not isinstance(rule['emission_type'], str) or not rule['emission_type'].strip()):
            raise ValueError('emission_type must be nonempty text')
        normalized.append(rule)
    return copy.deepcopy(dict(templates=templates, questions=normalized))


def definition(domain: str, domain_cfg: dict | None = None) -> dict:
    return validate_definition(domain_cfg['question_set'] if domain_cfg and 'question_set' in domain_cfg else _builtin(domain))


def questions_for_domain(domain: str, domain_cfg: dict | None = None) -> list[str]:
    return [q['prompt'] for q in definition(domain, domain_cfg)['questions']]


def question_rule(domain: str, index: int, domain_cfg: dict | None = None) -> dict:
    questions = definition(domain, domain_cfg)['questions']
    if type(index) is not int or not 1 <= index <= len(questions):
        raise ValueError('question index is outside the configured list')
    return questions[index - 1]


def emission_type_for_question(domain: str, question_index: int, domain_cfg: dict | None = None):
    return question_rule(domain, question_index, domain_cfg).get('emission_type')


def wildfire_question_EFQuery(emission: str) -> str:
    return _builtin('wildfire')['legacy']['wildfire_question_B_ef_details'].format(emission=emission)


def __getattr__(name: str):
    """Preserve legacy imports while keeping every research prompt outside Python."""
    templates = {'message_system_v2': 'system', 'message_system_v2_output': 'instruction',
                 'message_system_ragQuery': 'verifier_system', 'checker_requery': 'verifier_instruction',
                 'checker_requeryFull': 'verifier_full'}
    if name in templates:
        return _builtin('aqua')['templates'][templates[name]]
    questions = {'aqua_question_A_meta': ('aqua', 0), 'aqua_question_B_experiment': ('aqua', 1),
                 'aqua_question_C_flux': ('aqua', 2), 'wildfire_question_A_meta': ('wildfire', 0)}
    if name in questions:
        domain, index = questions[name]
        return _builtin(domain)['questions'][index]['prompt']
    if name == 'wildfire_question_B_ef_details':
        return _builtin('wildfire')['legacy'][name]
    raise AttributeError(name)
