import torch
from transformers import pipeline

class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = None

    def setup_generator(self):
        """Setup text generation pipeline."""
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return True, "Generator setup successfully."
        except Exception as e:
            return False, f"Failed to setup generator: {str(e)}"

    def _apply_seed(self, seed):
        try:
            if seed is None:
                return
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def generate_text(self, prompt, max_length=64, temperature=0.2, top_p=0.8, repetition_penalty=1.1, seed=None):
        """Generate text from prompt."""
        if not self.generator:
            success, msg = self.setup_generator()
            if not success:
                return msg

        self._apply_seed(seed)
        try:
            do_sample = temperature > 0 or top_p < 1.0
            effective_temp = max(0.0, float(temperature))
            effective_top_p = max(0.0, min(float(top_p), 1.0))
            rep_penalty = max(0.8, float(repetition_penalty))
            outputs = self.generator(
                prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=effective_temp if do_sample else 0.0,
                top_p=effective_top_p if do_sample else 1.0,
                repetition_penalty=rep_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = outputs[0]['generated_text']
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        except Exception as e:
            return f"Generation failed: {str(e)}"

    def test_model(self, test_prompts=None, **gen_kwargs):
        """Test model with sample prompts."""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Tell me a joke.",
            ]

        results = {}
        for prompt in test_prompts:
            result = self.generate_text(prompt, **gen_kwargs)
            results[prompt] = result

        return results
