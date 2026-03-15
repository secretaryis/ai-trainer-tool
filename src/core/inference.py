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
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return True, "Generator setup successfully."
        except Exception as e:
            return False, f"Failed to setup generator: {str(e)}"

    def generate_text(self, prompt, max_length=50):
        """Generate text from prompt."""
        if not self.generator:
            success, msg = self.setup_generator()
            if not success:
                return msg

        try:
            outputs = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = outputs[0]['generated_text']
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        except Exception as e:
            return f"Generation failed: {str(e)}"

    def test_model(self, test_prompts=None):
        """Test model with sample prompts."""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Tell me a joke.",
            ]

        results = {}
        for prompt in test_prompts:
            result = self.generate_text(prompt)
            results[prompt] = result

        return results