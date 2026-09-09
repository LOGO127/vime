"""Rollout batches preserve the sample stream across epoch boundaries."""

import json
from types import SimpleNamespace

import pytest

from vime.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer
from vime.utils.data import Dataset


def make_source(tmp_path, size, shuffle, source_type=RolloutDataSource):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(size)))
    source = source_type(
        SimpleNamespace(
            rollout_global_dataset=False,
            prompt_data=None,
            rollout_shuffle=shuffle,
            n_samples_per_prompt=2,
            buffer_filter_path=None,
        )
    )
    # Supply a real local Dataset; tokenizer/model loading is deliberately disabled.
    source.dataset = Dataset(str(path), tokenizer=None, processor=None, max_length=None, prompt_key="text", seed=42)
    if shuffle:
        source.dataset.shuffle(0)
    return source


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 6, 8, 11])
def test_batch_matches_single_sample_stream(tmp_path, shuffle, count):
    batched = make_source(tmp_path, 3, shuffle)
    singles = make_source(tmp_path, 3, shuffle)
    expected = [singles.get_samples(1)[0][0].prompt for _ in range(count)]
    result = batched.get_samples(count)
    actual = [group[0].prompt for group in result]
    assert len(result) == count
    assert actual == expected
    assert batched.sample_offset == singles.sample_offset
    assert batched.epoch_id == singles.epoch_id
    assert [sample.index for group in result for sample in group] == list(range(count * 2))
    assert batched.get_samples(1)[0][0].prompt == singles.get_samples(1)[0][0].prompt


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_after_multiple_epochs(tmp_path, shuffle):
    source = make_source(tmp_path, 3, shuffle)
    source.args.rollout_global_dataset = True
    source.args.save = str(tmp_path)
    source.get_samples(8)
    source.save(4)
    resumed = make_source(tmp_path, 3, shuffle)
    resumed.args.rollout_global_dataset = True
    resumed.args.load = str(tmp_path)
    resumed.load(4)
    expected = source.get_samples(8)
    actual = resumed.get_samples(8)
    assert [(g[0].prompt, g[0].group_index, g[0].index) for g in actual] == [
        (g[0].prompt, g[0].group_index, g[0].index) for g in expected
    ]
    assert len(actual) == 8


def test_empty_dataset_rejects_positive_request(tmp_path):
    source = make_source(tmp_path, 0, False)
    assert source.get_samples(0) == []
    with pytest.raises(ValueError, match="empty rollout dataset"):
        source.get_samples(1)


def test_repeated_prompts_are_independent_copies(tmp_path):
    source = make_source(tmp_path, 1, False)
    groups = source.get_samples(4)
    assert len(groups) == 4
    groups[0][0].metadata["changed"] = True
    assert groups[0][1].metadata == {}
    assert all(group[0].metadata == {} for group in groups[1:])
    assert source.dataset[0].metadata == {}


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("size", [1, 3, 7])
def test_sequential_batches_preserve_cursor(tmp_path, shuffle, size):
    source = make_source(tmp_path, size, shuffle)
    reference = make_source(tmp_path, size, shuffle)
    for count in (1, size, 0, size * 3 + 2, 2, size * 2):
        actual = source.get_samples(count)
        expected = [reference.get_samples(1)[0] for _ in range(count)]
        assert [(g[0].prompt, g[0].group_index, g[0].index) for g in actual] == [
            (g[0].prompt, g[0].group_index, g[0].index) for g in expected
        ]
        assert source.sample_offset == reference.sample_offset
        assert source.epoch_id == reference.epoch_id


@pytest.mark.parametrize("shuffle", [False, True])
def test_buffered_groups_precede_multi_epoch_top_up(tmp_path, shuffle):
    source = make_source(tmp_path, 3, shuffle, RolloutDataSourceWithBuffer)
    reference = make_source(tmp_path, 3, shuffle)
    buffered = source.get_samples(2)
    reference.get_samples(2)
    source.add_samples(buffered)
    expected = buffered + [reference.get_samples(1)[0] for _ in range(9)]
    actual = source.get_samples(11)
    assert len(actual) == 11
    assert source.get_buffer_length() == 0
    assert actual[:2] == buffered
    assert [(g[0].prompt, g[0].group_index, g[0].index) for g in actual] == [
        (g[0].prompt, g[0].group_index, g[0].index) for g in expected
    ]
    assert source.sample_offset == reference.sample_offset
    assert source.epoch_id == reference.epoch_id
