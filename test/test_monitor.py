import asyncio

import pytest

from lib.monitor import aggregate_scores, ensemble_monitor, parse_suspiciousness


def test_int_value():
    assert parse_suspiciousness('{"suspiciousness": 7}') == 7


def test_string_value_coerced():
    assert parse_suspiciousness('{"suspiciousness": "3", "reason": "x"}') == 3


def test_fenced_json():
    assert parse_suspiciousness('```json\n{"suspiciousness": 6}\n```') == 6


def test_aggregate_avg():
    assert aggregate_scores([2, 4, 6], "avg") == 4


def test_aggregate_max():
    assert aggregate_scores([2, 4, 6], "max") == 6


def test_aggregate_unknown_method():
    with pytest.raises(ValueError):
        aggregate_scores([1, 2], "median")


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, scores_by_model):
        self.scores_by_model = scores_by_model

    async def create(self, model, messages, response_format, temperature):
        return _FakeResponse(f'{{"suspiciousness": {self.scores_by_model[model]}}}')


class _FakeChat:
    def __init__(self, scores_by_model):
        self.completions = _FakeCompletions(scores_by_model)


class _FakeClient:
    def __init__(self, scores_by_model):
        self.chat = _FakeChat(scores_by_model)


def test_ensemble_monitor_avg():
    client = _FakeClient({"m1": 2, "m2": 8})
    result = asyncio.run(
        ensemble_monitor(client, ["m1", "m2"], "{task} {output}", "t", "o", "avg")
    )
    assert result == 5


def test_ensemble_monitor_max():
    client = _FakeClient({"m1": 2, "m2": 8})
    result = asyncio.run(
        ensemble_monitor(client, ["m1", "m2"], "{task} {output}", "t", "o", "max")
    )
    assert result == 8
