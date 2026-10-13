import numpy as np
import torch


def test_active_lengths_stop_at_first_special_token():
    from experiments.recovery_help.backend import active_lengths
    ids = torch.tensor([[4, 5, 2, 2], [6, 2, 2, 2], [7, 8, 9, 10]])
    assert active_lengths(ids, [2]).tolist() == [2, 1, 4]


def test_generation_capture_matches_normal_generation_and_cleans_hooks():
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from experiments.recovery_help.backend import generate_batch
    torch.manual_seed(13)
    tokenizer = Tokenizer(WordLevel({'[UNK]':0, '[PAD]':1, 'hello':2, 'world':3, 'again':4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]')
    tok.padding_side = 'left'
    config = LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, eos_token_id=None, pad_token_id=1)
    model = LlamaForCausalLM(config).eval()
    prompts = ['hello', 'hello world again']
    encoded = tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt')
    expected = model.generate(**encoded, max_new_tokens=4, do_sample=False, pad_token_id=1)
    result = generate_batch(model, tok, prompts, 4, capture=True)
    assert result['ids'] == expected[:, encoded.input_ids.shape[1]:].tolist()
    assert result['states'].shape == (2, 3, 2, 16)
    assert np.isfinite(result['states']).all() and np.isfinite(result['confidence']).all()
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)


def test_early_finished_rows_use_their_own_last_decision_state(monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, LogitsProcessor
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from experiments.recovery_help.backend import generate_batch
    tokenizer = Tokenizer(WordLevel({'[UNK]':0, '[PAD]':1, 'hello':2, 'world':3, '[EOS]':4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tok.padding_side = 'left'
    model = LlamaForCausalLM(LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, eos_token_id=4, pad_token_id=1)).eval()
    original = model.generate
    class Forced(LogitsProcessor):
        def __call__(self, ids, scores):
            t = ids.shape[1]-2
            scores[:] = -torch.inf
            scores[0, 4 if t>=2 else 2+t] = 0
            scores[1, 4 if t>=4 else 2+t%2] = 0
            return scores
    monkeypatch.setattr(model, 'generate', lambda **kw: original(**kw, logits_processor=[Forced()]))
    events = []
    handle = model.model.layers[2].register_forward_hook(lambda m,i,o: events.append(o[:, -1].detach().half().cpu()))
    result = generate_batch(model, tok, ['hello', 'hello world'], 6, capture=True)
    handle.remove()
    assert result['lengths'] == [2, 4]
    assert np.array_equal(result['states'][0,2,1], events[1][0].numpy())
    assert np.array_equal(result['states'][1,2,1], events[3][1].numpy())
    assert np.isfinite(result['confidence']).all()
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)
