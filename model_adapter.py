# model_adapter.py

"""
Generic model adapter for plug-and-play support with different LLM architectures.
Handles LLaMA, GPT, Mistral, and other decoder-only models.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ModelAdapter:
    """
    Adapter class that provides a unified interface for different LLM architectures.
    Handles model-specific differences in accessing hidden states and logits.
    """
    
    def __init__(self, model, model_name: str = "", device_name: str = ""):
        """
        Initialize the adapter with a model.
        
        Args:
            model: The loaded transformer model
            model_name: Name or path of the model (for architecture detection)
        """
        self.model = model
        self.model_name = model_name.lower()
        self.model_type = self._detect_model_type()
        self._logit_head = self._get_logit_head()
        self.device_name = device_name
        
    def _detect_model_type(self) -> str:
        """Detect the model architecture type for hallucination mitigation."""
        model_class_name = self.model.__class__.__name__.lower()
        
        # Check model class name first
        if 'llama' in model_class_name or 'llama' in self.model_name:
            return 'llama'
        elif 'mistral' in model_class_name or 'mistral' in self.model_name:
            return 'mistral'
        elif 'gpt' in model_class_name or 'gpt' in self.model_name:
            return 'gpt'
        elif 'phi' in model_class_name or 'phi' in self.model_name:
            return 'phi'
        elif 'qwen' in model_class_name or 'qwen' in self.model_name:
            return 'qwen'
        elif 'gemma' in model_class_name or 'gemma' in self.model_name:
            return 'gemma'
        elif 'falcon' in model_class_name or 'falcon' in self.model_name:
            return 'falcon'
        elif 'mpt' in model_class_name or 'mpt' in self.model_name:
            return 'mpt'
        elif 'bloom' in model_class_name or 'bloom' in self.model_name:
            return 'bloom'
        elif 'opt' in model_class_name or 'opt' in self.model_name:
            return 'opt'
        else:
            # Try to detect from model structure
            if hasattr(self.model, 'lm_head'):
                return 'llama_like'
            elif hasattr(self.model, 'embed_out'):
                return 'gpt_like'
            elif hasattr(self.model, 'output_projection'):
                return 'custom'
            else:
                return 'generic'
    
    def _get_logit_head(self):
        """Get the logit head/language model head from the model."""
        # Try common attribute names
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out
        elif hasattr(self.model, 'head'):
            return self.model.head
        elif hasattr(self.model, 'output_projection'):
            return self.model.output_projection
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'lm_head'):
            return self.model.language_model.lm_head
        else:
            # For models without explicit head, use the embedding layer
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                # GPT-style: use embedding weights
                return lambda x: F.linear(x, self.model.transformer.wte.weight)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # LLaMA-style: use embedding weights
                return lambda x: F.linear(x, self.model.model.embed_tokens.weight)
            else:
                raise AttributeError(
                    f"Could not find logit head in model. Model type: {self.model_type}, "
                    f"Model class: {self.model.__class__.__name__}"
                )
    
    def get_logits_from_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Get logits from a hidden state tensor.
        
        Args:
            hidden_state: Hidden state tensor of shape [batch, seq_len, hidden_dim]
            
        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        if callable(self._logit_head):
            return self._logit_head(hidden_state)
        else:
            # If it's a module, call it
            return self._logit_head(hidden_state)
    
    def forward_with_hidden_states(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that returns both logits and hidden states.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Tuple of (logits, hidden_states_list)
        """
        # Try to get hidden states
        try:
            outputs = self.model(input_ids, output_hidden_states=True)
            
            # Extract hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
            elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                # For encoder-decoder models
                hidden_states = outputs.decoder_hidden_states
            else:
                # Fallback: extract from model layers manually
                hidden_states = self._extract_hidden_states_manual(input_ids)
            
            # Get final logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # Compute from last hidden state
                last_hidden = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
                logits = self.get_logits_from_hidden_state(last_hidden)
            
            return logits, hidden_states
            
        except Exception as e:
            # Fallback for models that don't support output_hidden_states
            print(f"Warning: Could not get hidden states directly: {e}. Using fallback method.", flush=True)
            return self._forward_fallback(input_ids)
    
    def _extract_hidden_states_manual(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Manually extract hidden states by hooking into model layers."""
        hidden_states = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states.append(output[0])
            else:
                hidden_states.append(output)
        
        # Try to find transformer layers (for hallucination mitigation)
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'blocks'):
            layers = self.model.transformer.blocks
        
        if layers is None:
            # Can't extract hidden states
            raise ValueError("Could not find model layers to extract hidden states")
        
        # Register hooks
        hooks = []
        for layer in layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        try:
            # Forward pass
            _ = self.model(input_ids)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return hidden_states
    
    def _forward_fallback(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Fallback forward pass when hidden states aren't directly available."""
        outputs = self.model(input_ids)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # Last resort: use model output directly
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                raise ValueError("Could not extract logits from model output")
        
        # Create a dummy hidden states list (just the final layer)
        # This allows the code to work but without layer-level access
        with torch.no_grad():
            # Get embeddings
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                hidden = self.model.model.embed_tokens(input_ids)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                hidden = self.model.transformer.wte(input_ids)
            else:
                # Can't get hidden states
                hidden = None
            
            if hidden is not None:
                hidden_states = [hidden]
            else:
                # Empty list - will cause fallback in generation methods
                hidden_states = []
        
        return logits, hidden_states
    
    def get_num_layers(self) -> Optional[int]:
        """Get the number of layers in the model."""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'num_layers'):
            return self.model.config.num_layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return len(self.model.gpt_neox.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'blocks'):
            return len(self.model.transformer.blocks)
        else:
            return None
    
    def supports_layer_access(self) -> bool:
        """Check if the model supports layer-level hidden state access."""
        try:
            test_input = torch.tensor([[1, 2, 3]], device=self.device_name)
            _, hidden_states = self.forward_with_hidden_states(test_input)
            return len(hidden_states) > 0
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return False

