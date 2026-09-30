"""Judge provider protocol."""
from __future__ import annotations

from typing import Protocol

from ..schemas import EvalSample, JudgeOutput


class JudgeProvider(Protocol):
    name: str

    def judge(self, sample: EvalSample) -> JudgeOutput: ...
