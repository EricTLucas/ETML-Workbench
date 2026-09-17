from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import uuid


def json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Step:
    operation: str
    columns: tuple[str, ...]
    params: dict = field(default_factory=dict)
    reason: str = ''
    evidence: dict = field(default_factory=dict)
    enabled: bool = True
    status: str = 'proposed'
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    approval_fingerprint: str | None = None

    def __post_init__(self):
        if isinstance(self.columns, str):
            raise ValueError('columns must be a sequence, not a string')
        object.__setattr__(self, 'columns', tuple(self.columns))
        if not self.columns or not all(isinstance(c, str) and c for c in self.columns):
            raise ValueError('Provide nonempty column names')
        if len(set(self.columns)) != len(self.columns):
            raise ValueError('Duplicate step columns')
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError('operation must be a string')
        if type(self.enabled) is not bool or self.status not in {'proposed', 'approved', 'rejected'}:
            raise ValueError('Invalid step status/enabled value')
        if not isinstance(self.step_id, str) or not self.step_id:
            raise ValueError('step_id must be nonempty')
        if not isinstance(self.params, dict) or not isinstance(self.evidence, dict):
            raise ValueError('params and evidence must be dictionaries')
        object.__setattr__(self, 'params', json_copy(self.params))
        object.__setattr__(self, 'evidence', json_copy(self.evidence))

    def specification(self):
        return {'operation': self.operation, 'columns': list(self.columns), 'params': self.params}

    @property
    def approved(self):
        return self.status == 'approved' and self.approval_fingerprint == fingerprint(self.specification())

    def approve(self):
        return replace(self, status='approved', approval_fingerprint=fingerprint(self.specification()))

    def reject(self):
        return replace(self, status='rejected', enabled=False, approval_fingerprint=None)

    def edit(self, **changes):
        changes.update(status='proposed', approval_fingerprint=None)
        return replace(self, **changes)

    def to_dict(self):
        return json_copy({**self.specification(), 'reason': self.reason, 'evidence': self.evidence,
                          'enabled': self.enabled, 'status': self.status, 'step_id': self.step_id,
                          'approval_fingerprint': self.approval_fingerprint})


@dataclass(frozen=True)
class Recipe:
    steps: tuple[Step, ...] = ()
    name: str = 'Preprocessing'
    format_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'steps', tuple(self.steps))
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError('Unsupported recipe version')
        if not all(isinstance(step, Step) for step in self.steps):
            raise ValueError('Recipe steps must be Step objects')
        if len({s.step_id for s in self.steps}) != len(self.steps):
            raise ValueError('Duplicate step IDs')

    @property
    def active_steps(self):
        return tuple(s for s in self.steps if s.enabled and s.status != 'rejected')

    @property
    def fingerprint(self):
        return fingerprint([s.specification() for s in self.active_steps])

    def approve_all(self):
        return replace(self, steps=tuple(s.approve() if s.enabled and s.status != 'rejected' else s
                                         for s in self.steps))

    def to_dict(self):
        return {'format_version': self.format_version, 'name': self.name,
                'steps': [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, payload):
        value = json_copy(payload)
        value['steps'] = tuple(Step(**s) for s in value.get('steps', []))
        return cls(**value)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False, allow_nan=False)
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
