"""
Model wrapper for language models (OpenAI API).
"""
import os
from typing import Optional
from omegaconf import DictConfig
import openai


class OpenAIModel:
    """Wrapper for OpenAI API models."""
    
    def __init__(self, model_cfg: DictConfig):
        """
        Initialize OpenAI model.
        
        Args:
            model_cfg: Model configuration with name, provider, etc.
        """
        self.model_name = model_cfg.name
        self.max_tokens = model_cfg.get('max_tokens', 512)
        
        # Initialize OpenAI client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("[model] WARNING: OPENAI_API_KEY not set, using mock responses")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        if self.client is None:
            # Mock response for testing when API key not available
            return self._mock_generate(prompt, temperature)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"[model] Error calling OpenAI API: {e}")
            print(f"[model] Falling back to mock response")
            return self._mock_generate(prompt, temperature)
    
    def _mock_generate(self, prompt: str, temperature: float) -> str:
        """
        Generate mock response for testing.
        Returns a plausible chain-of-thought solution.
        """
        # Check if this is a paraphrase request
        if "Paraphrase" in prompt or "paraphrase" in prompt:
            return """1. The problem can be restated as follows: what is the solution?
2. Rephrasing the question: how can we solve this?
3. In other words: what is the answer?"""
        
        # Otherwise, generate a mock math solution
        return """Let me solve this step by step:

Step 1: First, I'll identify the key information in the problem.
Step 2: Next, I'll set up the equation based on the problem.
Step 3: Now I'll solve for the unknown variable.
Step 4: Finally, I'll verify the answer makes sense.

The answer is: #### 42"""
