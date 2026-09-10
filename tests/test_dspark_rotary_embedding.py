import math

import torch

from vime.backends.megatron_utils.dspark.attention import DSparkParallelAttention, DSparkRotaryEmbedding


NUM_GPUS = 0


def test_dspark_rotary_embedding_keeps_long_position_ids_out_of_output_dtype():
    rotary = DSparkRotaryEmbedding(head_dim=8, rotary_base=10000.0)
    position_ids = torch.tensor([[0, 1, 17, 128]], dtype=torch.long)

    cos, sin = rotary(position_ids)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    freqs = torch.einsum("i,bj->bji", inv_freq, position_ids.float())
    emb = torch.cat([freqs, freqs], dim=-1)
    expected_cos = emb.cos().unsqueeze(1)
    expected_sin = emb.sin().unsqueeze(1)

    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0)
    assert torch.count_nonzero(sin).item() > 0
    assert math.isclose(float(cos.abs().max()), 1.0, rel_tol=0, abs_tol=1e-7)


def test_dspark_attention_casts_rotary_values_to_query_dtype():
    attention = DSparkParallelAttention(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    ).to(dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 3, 256, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(2, 5, 256, dtype=torch.bfloat16)
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    attention_mask = torch.ones((2, 1, 3, 8), dtype=torch.bool)

    output = attention(
        hidden_states=hidden_states,
        target_hidden_states=target_hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    assert output.dtype == torch.bfloat16
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
