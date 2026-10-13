from types import SimpleNamespace
import pytest


def test_fast_recovery_counts_only_nonstop_tokens_and_preserves_input_lengths():
    from experiments.recovery_scale import backend
    assert hasattr(backend,'unpack_fast'), 'Fast recovery output conversion is missing'
    outputs=[SimpleNamespace(prompt_token_ids=[10,11,12],outputs=[SimpleNamespace(text='ok',token_ids=[5,6,2])]),
             SimpleNamespace(prompt_token_ids=[10],outputs=[SimpleNamespace(text='long',token_ids=[4,5,6])])]
    result=backend.unpack_fast(outputs,3,{2})
    assert result['lengths']==[2,3]
    assert result['at_cap']==[False,True]
    assert result['input_lengths']==[3,1]
