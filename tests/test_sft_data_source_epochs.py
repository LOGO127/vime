"""Offline SFT data-path regression; no model inference or trainer execution."""

import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vime.rollout import sft_rollout
from vime.rollout.data_source import RolloutDataSource


@pytest.mark.parametrize("shuffle", [False, True])
def test_sft_batch_crosses_epochs_and_restores_cursor(tmp_path, monkeypatch, shuffle):
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "user": 1, "assistant": 2, "0": 3, "1": 4, "2": 5, "answer": 6},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
    )
    model_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model_path)
    conversations = [
        [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "answer"}] for i in range(3)
    ]
    data_path = tmp_path / "prompts.jsonl"
    data_path.write_text("".join(json.dumps({"text": row}) + "\n" for row in conversations), encoding="utf-8")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=str(data_path),
        hf_checkpoint=str(model_path),
        dump_details=None,
        rollout_max_prompt_len=None,
        input_key="text",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        rollout_shuffle=shuffle,
        n_samples_per_prompt=1,
        rollout_batch_size=8,
        loss_mask_type="qwen3_5",
        save=str(tmp_path / "checkpoint"),
        load=str(tmp_path / "checkpoint"),
    )
    # Reset only process-global caches; production loaders and mask generation remain real.
    for name in ("TOKENIZER", "PROCESSOR", "MASK_GENERATOR"):
        monkeypatch.setattr(sft_rollout, name, None)
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", False)
    source = RolloutDataSource(args)
    reference = RolloutDataSource(args)
    expected = [reference.get_samples(1)[0][0].prompt for _ in range(8)]
    groups = sft_rollout.generate_rollout(args, 0, source)
    print(
        {
            "shuffle": shuffle,
            "requested": 8,
            "sft_returned": len(groups),
            "epoch": source.epoch_id,
            "offset": source.sample_offset,
        }
    )
    assert len(groups) == 8
    assert [group[0].prompt for group in groups] == expected
    for (sample,) in groups:
        assert sample.tokens
        assert 0 < sample.response_length <= len(sample.tokens)
        assert len(sample.loss_mask) == sample.response_length
        assert any(sample.loss_mask)
    source.save(0)
    resumed = RolloutDataSource(args)
    resumed.load(0)
    continued = sft_rollout.generate_rollout(args, 1, source)
    restored = sft_rollout.generate_rollout(args, 1, resumed)
    assert len(restored) == 8
    assert [(g[0].index, g[0].prompt, g[0].tokens, g[0].loss_mask) for g in restored] == [
        (g[0].index, g[0].prompt, g[0].tokens, g[0].loss_mask) for g in continued
    ]
