import pytest

pytest.importorskip("torch")

from core.inference import InferenceEngine


class DummyGen:
    def __init__(self):
        self.called_with = None

    def __call__(self, prompt, **kwargs):
        self.called_with = kwargs
        return [{"generated_text": f"{prompt} world"}]


class DummyTokenizer:
    eos_token_id = 0


def test_generate_text_strips_prompt_and_uses_defaults():
    engine = InferenceEngine(model=None, tokenizer=DummyTokenizer())
    dummy = DummyGen()
    engine.generator = dummy
    out = engine.generate_text("hello", max_length=10, temperature=0.0, top_p=1.0, repetition_penalty=1.1, seed=123)
    assert out.strip() == "world"
    assert dummy.called_with["do_sample"] is False
    assert dummy.called_with["max_new_tokens"] == 10
