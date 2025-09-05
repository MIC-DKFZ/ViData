from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "data",
    [
        {"string": "string", "int": 42, "float": 42.2, "bool": True},
        {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}},
        {"nested1": {"a": [1, 2, 3]}, "nested2": [{"a": 1, "b": 2}]},
    ],
)
def test_yaml(data, tmp_path):
    from vidata.io import load_yaml, save_yaml

    file = Path(tmp_path).joinpath("data.yaml")

    save_yaml(data, file)
    data_l = load_yaml(file)

    assert data == data_l


@pytest.mark.parametrize(
    "data",
    [
        {"string": "string", "int": 42, "float": 42.2, "bool": True},
        {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}},
        {"nested1": {"a": [1, 2, 3]}, "nested2": [{"a": 1, "b": 2}]},
    ],
)
def test_json(data, tmp_path):
    from vidata.io import load_json, save_json

    file = Path(tmp_path).joinpath("data.json")

    save_json(data, file)
    data_l = load_json(file)

    assert data == data_l


@pytest.mark.parametrize(
    "data",
    [
        {"string": "string", "int": 42, "float": 42.2, "bool": True},
        {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}},
        {"nested1": {"a": [1, 2, 3]}, "nested2": [{"a": 1, "b": 2}]},
    ],
)
def test_pickle(data, tmp_path):
    from vidata.io import load_pickle, save_pickle

    file = Path(tmp_path).joinpath("data.pkl")

    save_pickle(data, file)
    data_l = load_pickle(file)

    assert data == data_l
