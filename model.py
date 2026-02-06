# model.py

import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ReadTimeoutError, NameResolutionError, MaxRetryError
from model_adapter import ModelAdapter
from utils import clear_cuda_cache


class LLamaQaStoppingCriteria(StoppingCriteria):
    """
    Stops generation when the sequence ends with any of the provided stop token sequences.

    stop_words_ids_list: list of token id sequences (each is a List[int])
    """

    def __init__(self, stop_words_ids_list):
        super().__init__()
        self.stop_sequences = stop_words_ids_list or []

    def _ends_with_sequence(self, sequence_tensor, stop_sequence):
        if len(stop_sequence) == 0:
            return False
        if sequence_tensor.shape[-1] < len(stop_sequence):
            return False
        tail = sequence_tensor[0, -len(stop_sequence):].tolist()
        return tail == stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids is None or len(self.stop_sequences) == 0:
            return False
        # If any sequence in the batch ends with any stop sequence, stop
        for stop_sequence in self.stop_sequences:
            if self._ends_with_sequence(input_ids, stop_sequence):
                return True
        return False

# We'll import the JSD function from utils:
from utils import jensen_shannon_divergence

class Base_Model:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=24):
        """
        Initializes the model, tokenizer, and related configurations.
        """
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.stop_word_ids = []
        self.stopping_criteria = None

        self.model, self.tokenizer = self.load_model(model_name)
        # Clear cache after model loading
        clear_cuda_cache()
        # Initialize model adapter for plug-and-play hallucination mitigation
        self.adapter = ModelAdapter(self.model, model_name, self.device)
        self.supports_layers = self.adapter.supports_layer_access()
        num_layers = self.adapter.get_num_layers()
        print(f"Model adapter initialized. Model type: {self.adapter.model_type}, "
              f"Supports layer access: {self.supports_layers}, "
              f"Number of layers: {num_layers}", flush=True)
        # Clear cache after adapter initialization
        clear_cuda_cache()

    def load_model(self, model_name):
        """
        Loads the tokenizer and model given the model_name.
        Configures device mapping and GPU memory usage if running on CUDA.
        """
        # Resolve device and dtype safely
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.", flush=True)
            self.device = "cpu"
            

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

        if self.device == "cuda":
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        elif self.device != "cpu":
            raise ValueError(f"Invalid device: {self.device}")

        # If the model name contains 'vicuna', we assume a different tokenizer path
        tokenizer_name = model_name if 'vicuna' not in model_name else 'huggyllama/llama-7b'

        # Detect local path usage (to avoid any network calls)
        is_local_path = os.path.isdir(model_name)

        def _disable_hf_transfer():
            # Avoid xet-bridge.hf.co DNS by disabling hf_transfer fast downloader
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

        # Try online first unless local path provided, then fall back to offline
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                local_files_only=is_local_path,
            )
        except (RequestsConnectionError, ReadTimeoutError, NameResolutionError, MaxRetryError, OSError) as e:
            _disable_hf_transfer()
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                local_files_only=True,
            )

        # Prepare model kwargs - use bfloat16 for better numerical stability
        model_kwargs = kwargs.copy()
        model_kwargs.pop('torch_dtype', None)  # Remove torch_dtype from kwargs to avoid conflict
        model_kwargs['torch_dtype'] = torch.bfloat16  # Use bfloat16 explicitly
        
        # Try with flash-attention-2 first (faster and more memory efficient)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=is_local_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                use_cache=True,
                attn_implementation="flash_attention_2"
            )
            print(model.config._attn_implementation)
        except (ValueError, ImportError, AttributeError, TypeError) as e:
            # Flash-attention not available or incompatible - fall back to standard attention
            error_str = str(e).lower()
            if ('flash' in error_str or 'attention' in error_str or 'flash_attn' in error_str or 
                'attn_implementation' in error_str or 'unexpected keyword argument' in error_str):
                print("Warning: flash-attn not available or incompatible. Falling back to standard attention.", flush=True)
                print("To use flash-attention, install with: pip install flash-attn --no-build-isolation", flush=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_files_only=is_local_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
            else:
                # Re-raise if it's a different error
                raise
        except (RequestsConnectionError, ReadTimeoutError, NameResolutionError, MaxRetryError, OSError) as e:
            _disable_hf_transfer()
            # Try with flash-attention first in fallback
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    local_files_only=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
            except (ValueError, ImportError, AttributeError, TypeError) as e2:
                # Flash-attention not available - fall back to standard attention
                error_str = str(e2).lower()
                if ('flash' in error_str or 'attention' in error_str or 'flash_attn' in error_str or 
                    'attn_implementation' in error_str or 'unexpected keyword argument' in error_str):
                    print("Warning: flash-attn not available or incompatible. Falling back to standard attention.", flush=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        local_files_only=True,
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                    )
                else:
                    raise

        self.device = "cuda" if torch.cuda.is_available() and self.num_gpus >= 1 else "cpu"
        if self.device == "cuda":
            # Clear cache before moving model to GPU
            clear_cuda_cache()
        self.model = model.to(self.device)
        # Set model to eval mode for memory efficiency
        self.model.eval()
        if self.device == "cuda":
            # Clear cache after moving model to GPU
            clear_cuda_cache()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        """
        Sets up stopping criteria for the model based on a list of stop words.
        Each stop word is tokenized and appended to stop_word_ids.
        """
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()

        for stop_word in self.stop_words:
            # Encode "\n<stop_word>" and skip the first 3 tokens
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            self.stop_word_ids.extend(stop_word_ids)
            print(f"Added stop word: {stop_word} with the ids {stop_word_ids}", flush=True)

        stop_words_ids_list = [self.tokenizer.encode('\n' + word)[3:] for word in stop_words]
        self.stopping_criteria.append(LLamaQaStoppingCriteria(stop_words_ids_list))

    def _get_max_input_length(self, max_new_tokens=20):
        """
        Gets the maximum input length for tokenization, reserving space for generation.
        Returns the model's max sequence length minus reserved tokens for generation.
        """
        # Try to get max length from model config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'max_position_embeddings'):
                max_length = self.model.config.max_position_embeddings
            elif hasattr(self.model.config, 'max_seq_length'):
                max_length = self.model.config.max_seq_length
            else:
                max_length = None
        else:
            max_length = None
        
        # Fallback to tokenizer's model_max_length
        if max_length is None:
            if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length:
                max_length = self.tokenizer.model_max_length
            else:
                # Default fallback
                max_length = 2048
        
        # Reserve tokens for generation (max_new_tokens + some buffer)
        reserved = max_new_tokens + 50
        max_input_length = max_length - reserved
        
        # Ensure we have at least some reasonable minimum
        return max(100, max_input_length)

    #****************************************************
    # Modified By: Syed Kamal Ahmed Hiron (IIT Patna)
    #****************************************************
    
    def generate(
        self,
        input_text,
        input_text2,
        mode='final_layer_context',
        alpha=0.25,
        layer_alpha=0.5,
        start_layer=16,
        subset_layers=None,
        max_new_tokens=20
    ):
        """
        Generates text based on the specified mode. The advanced modes include:
          - 'final_layer_no_context'
          - 'final_layer_context'
          - 'CAD'
          - 'DOLA'
          - 'LACD'
          - 'ALACD'
          - 'contrast_layer_context_nocontext_jsd_subset'
        and so on.

        For each mode, calls the appropriate private method. 
        """
        # Use inference_mode for better memory efficiency
        with torch.inference_mode():
            # Auto-detect start_layer if not provided
            if start_layer is None:
                num_layers = self.adapter.get_num_layers()
                if num_layers:
                    start_layer = int(num_layers * 0.75)
                else:
                    start_layer = 16  # Fallback
            
            # Check if layer-based modes are supported (for hallucination mitigation)
            layer_based_modes = ['DOLA', 'contrast_layer_context_nocontext_jsd', 
                                'contrast_layer_context_nocontext_jsd_subset']
            '''if mode in layer_based_modes and not self.supports_layers:
                print(f"Warning: Mode '{mode}' requires layer access for hallucination mitigation, "
                      f"but model doesn't support it. Falling back to 'final_layer_context'.", flush=True)
                mode = 'final_layer_context' '''
            
            # Get max input length (reserving space for generation)
            max_input_length = self._get_max_input_length(max_new_tokens)
            
            # Tokenize input texts with truncation to prevent exceeding model's max length
            context_ids = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            ).input_ids.to(self.device)
            no_context_ids = self.tokenizer(
                input_text2,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            ).input_ids.to(self.device)

            initial_length_context = context_ids.shape[1]
            initial_length_no_context = no_context_ids.shape[1]

            # Map mode to methods
            mode_function_map = {
                # Baselines
                'final_layer_no_context': self._generate_final_layer_no_context,
                'final_layer_context': self._generate_final_layer_context,

                # CAD, DOLA
                'CAD': self._generate_CAD,
                'DOLA': self._generate_DOLA,

                # Contrast with JSD
                'LACD': self._generate_contrast_layer_adjusted_context_jsd,
                'ALACD': self._generate_advanced_contrast_layer_adjusted_context_jsd,
                'contrast_layer_context_nocontext_jsd_subset': self._generate_contrast_layer_adjusted_context_jsd_subset
            }

            # Prepare arguments
            mode_args = {
                'final_layer_no_context': (no_context_ids, initial_length_no_context, max_new_tokens),
                'final_layer_context': (context_ids, initial_length_context, max_new_tokens),
                'CAD': (context_ids, no_context_ids, max_new_tokens, initial_length_context, alpha, layer_alpha),
                'DOLA': (context_ids, no_context_ids, max_new_tokens, initial_length_context, alpha, layer_alpha, start_layer),
                'LACD': (
                    context_ids, no_context_ids, max_new_tokens,
                    initial_length_context, alpha, layer_alpha, start_layer
                ),
                'ALACD': (
                    context_ids, no_context_ids, max_new_tokens,
                    initial_length_context, alpha, layer_alpha, start_layer
                ),
                'contrast_layer_context_nocontext_jsd_subset': (
                    context_ids, no_context_ids, max_new_tokens,
                    initial_length_context, alpha, layer_alpha, subset_layers
                )
            }

            if mode not in mode_function_map:
                valid_modes = list(mode_function_map.keys())
                raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")

            # Call the corresponding method
            return mode_function_map[mode](*mode_args[mode])

    # --------------------------------------------------------------------------------
    # Baseline: final layer only
    # --------------------------------------------------------------------------------

    def _generate_final_layer_context(self, generated_context, initial_length_context, max_new_tokens):
        """
        Baseline - Only the context prompt
        """
        for _ in range(max_new_tokens):
            logits, hidden_states = self.adapter.forward_with_hidden_states(generated_context)
            if hidden_states:
                final_layer_logits_context = self.adapter.get_logits_from_hidden_state(hidden_states[-1][:, -1:, :])
                # Delete hidden states immediately to free memory
                del hidden_states
            else:
                final_layer_logits_context = logits[:, -1:, :]
            log_probs_final = F.log_softmax(final_layer_logits_context, dim=-1)

            next_token_id = torch.argmax(log_probs_final).unsqueeze(0)
            if next_token_id.item() in self.stop_word_ids:
                # Delete intermediate tensors before breaking
                del logits, final_layer_logits_context, log_probs_final
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            
            # Delete intermediate tensors and clear cache after each token
            del logits, final_layer_logits_context, log_probs_final
            if self.device == "cuda":
                clear_cuda_cache()

        new_tokens = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        # Clear cache after generation
        del generated_context
        if self.device == "cuda":
            clear_cuda_cache()
        return new_tokens

    def _generate_final_layer_no_context(self, generated_no_context, initial_length_no_context, max_new_tokens):
        """
        Baseline - Only the no-context prompt
        """
        for _ in range(max_new_tokens):
            logits, hidden_states = self.adapter.forward_with_hidden_states(generated_no_context)
            if hidden_states:
                final_layer_logits_no_context = self.adapter.get_logits_from_hidden_state(hidden_states[-1][:, -1:, :])
                # Delete hidden states immediately to free memory
                del hidden_states
            else:
                final_layer_logits_no_context = logits[:, -1:, :]
            log_probs_final = F.log_softmax(final_layer_logits_no_context, dim=-1)

            next_token_id = torch.argmax(log_probs_final).unsqueeze(0)
            if next_token_id.item() in self.stop_word_ids:
                # Delete intermediate tensors before breaking
                del logits, final_layer_logits_no_context, log_probs_final
                break

            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)
            
            # Delete intermediate tensors and clear cache after each token
            del logits, final_layer_logits_no_context, log_probs_final
            if self.device == "cuda":
                clear_cuda_cache()

        new_tokens = self.tokenizer.decode(
            generated_no_context[0, initial_length_no_context:], skip_special_tokens=True
        )
        # Clear cache after generation
        del generated_no_context
        if self.device == "cuda":
            clear_cuda_cache()
        return new_tokens

    # --------------------------------------------------------------------------------
    # CAD / DOLA
    # --------------------------------------------------------------------------------

    def _generate_CAD(self, context_ids, no_context_ids,
                      max_new_tokens, initial_length_context, alpha, layer_alpha):
        """
        CAD: (1+alpha)*A_final - alpha*B_final => greedy argmax
        """
        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            logits_ctx, hidden_states_ctx = self.adapter.forward_with_hidden_states(generated_context)
            logits_nocxt, hidden_states_nocxt = self.adapter.forward_with_hidden_states(generated_no_context)
            
            if hidden_states_ctx and hidden_states_nocxt:
                final_logits_context = self.adapter.get_logits_from_hidden_state(hidden_states_ctx[-1][:, -1:, :])
                final_logits_no_context = self.adapter.get_logits_from_hidden_state(hidden_states_nocxt[-1][:, -1:, :])
                # Delete hidden states immediately
                del hidden_states_ctx, hidden_states_nocxt
            else:
                final_logits_context = logits_ctx[:, -1:, :]
                final_logits_no_context = logits_nocxt[:, -1:, :]

            adjusted_logits = (1 + alpha)*final_logits_context - alpha*final_logits_no_context
            log_probs_final = F.log_softmax(adjusted_logits, dim=-1)

            next_token_id = torch.argmax(log_probs_final).unsqueeze(0)
            if next_token_id.item() in self.stop_word_ids:
                # Delete intermediate tensors before breaking
                del logits_ctx, logits_nocxt, final_logits_context, final_logits_no_context, adjusted_logits, log_probs_final
                break

            # Append the same token to both sequences
            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)
            
            # Delete intermediate tensors and clear cache after each token
            del logits_ctx, logits_nocxt, final_logits_context, final_logits_no_context, adjusted_logits, log_probs_final
            if self.device == "cuda":
                clear_cuda_cache()

        new_tokens = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        # Clear cache after generation
        del generated_context, generated_no_context
        if self.device == "cuda":
            clear_cuda_cache()
        return new_tokens

    def _generate_DOLA(self, context_ids, no_context_ids,
                       max_new_tokens, initial_length_context, alpha, layer_alpha, start_layer):
        """
        DOLA with JSD:
          1. Compute final-layer distribution
          2. Compare final-layer distribution to mid-layer distribution
             across layers >= start_layer, find max JSD
          3. Adjust final distribution with chosen mid-layer
        """
        generated_context = context_ids.clone()

        for _ in range(max_new_tokens):
            logits_ctx, hidden_states_ctx = self.adapter.forward_with_hidden_states(generated_context)

            # Final layer distribution
            if hidden_states_ctx:
                final_logits_context = self.adapter.get_logits_from_hidden_state(hidden_states_ctx[-1][:, -1:, :])
            else:
                final_logits_context = logits_ctx[:, -1:, :]
            probs_final = F.softmax(final_logits_context, dim=-1).squeeze()

            # Compare with each layer >= start_layer
            jsd_divergences = []
            mid_probs_list = []

            if hidden_states_ctx:
                for layer_idx, hidden_state in enumerate(hidden_states_ctx):
                    if layer_idx >= start_layer:
                        mid_logits_context = self.adapter.get_logits_from_hidden_state(hidden_state[:, -1:, :])
                        mid_probs = F.softmax(mid_logits_context, dim=-1).squeeze()

                        jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                        jsd_divergences.append((jsd_val.item(), layer_idx))
                        mid_probs_list.append(mid_probs)
                        # Delete intermediate tensors
                        del mid_logits_context, mid_probs, jsd_val

            # Delete hidden states after processing
            del hidden_states_ctx, logits_ctx

            # If no valid layers found, fallback to final
            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final).unsqueeze(0)
                del probs_final
            else:
                # Pick layer with maximum JSD
                _, max_jsd_index = max(jsd_divergences, key=lambda x: x[0])
                offset_idx = max_jsd_index - start_layer
                if offset_idx < 0:
                    offset_idx = 0
                chosen_mid_probs = mid_probs_list[offset_idx]

                if layer_alpha == 0:
                    idea_probs = probs_final - chosen_mid_probs
                else:
                    idea_probs = (1 + layer_alpha)*probs_final - layer_alpha*chosen_mid_probs

                next_token_id = torch.argmax(idea_probs).unsqueeze(0)
                # Delete intermediate tensors
                del probs_final, chosen_mid_probs, idea_probs, mid_probs_list, jsd_divergences

            if next_token_id.item() in self.stop_word_ids:
                del next_token_id
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            del next_token_id
            
            # Clear cache after each token
            if self.device == "cuda":
                clear_cuda_cache()

        # Return newly generated tokens
        new_tokens = self.tokenizer.decode(generated_context[0, initial_length_context:], skip_special_tokens=True)
        # Clear cache after generation
        del generated_context
        if self.device == "cuda":
            clear_cuda_cache()
        return new_tokens

    # --------------------------------------------------------------------------------
    # (A-B vs A-B) JSD
    # --------------------------------------------------------------------------------

    def _generate_contrast_layer_adjusted_context_jsd(
        self,
        context_ids,
        no_context_ids,
        max_new_tokens,
        initial_length_context,
        alpha,
        layer_alpha,
        start_layer
    ):
        """
        Compares final-layer (A-B) to mid-layer (A-B) for each layer >= start_layer,
        uses JSD to choose the best mid-layer, then adjusts the final distribution.
        """

        print(f"MODE: LACD")
        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            out_ctx = self.model(generated_context, output_hidden_states=True)
            out_nocxt = self.model(generated_no_context, output_hidden_states=True)

            final_logits_context = self.model.lm_head(out_ctx.hidden_states[-1][:, -1:, :])
            final_logits_no_context = self.model.lm_head(out_nocxt.hidden_states[-1][:, -1:, :])
            final_adjusted_logits = (1 + alpha)*final_logits_context - alpha*final_logits_no_context
            probs_final = F.softmax(final_adjusted_logits, dim=-1).squeeze()

            jsd_divergences = []
            mid_probs_list = []

            for layer_idx, (ctx_hid, nocxt_hid) in enumerate(
                zip(out_ctx.hidden_states, out_nocxt.hidden_states)
            ):
                if layer_idx >= start_layer:
                    mid_logits_ctx = self.model.lm_head(ctx_hid[:, -1:, :])
                    mid_logits_nocxt = self.model.lm_head(nocxt_hid[:, -1:, :])
                    mid_adjusted_logits = (1 + alpha)*mid_logits_ctx - alpha*mid_logits_nocxt
                    mid_probs = F.softmax(mid_adjusted_logits, dim=-1).squeeze()

                    jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                    jsd_divergences.append((jsd_val.item(), layer_idx))
                    mid_probs_list.append(mid_probs)

                    # Free per-layer logits to reduce peak memory
                    del mid_logits_ctx, mid_logits_nocxt, mid_adjusted_logits, jsd_val

            # We no longer need full hidden states from this step
            del out_ctx, out_nocxt, final_logits_context, final_logits_no_context, final_adjusted_logits

            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final).unsqueeze(0)
            else:
                _, max_jsd_idx = max(jsd_divergences, key=lambda x: x[0])
                offset_idx = max_jsd_idx - start_layer
                if offset_idx < 0:
                    offset_idx = 0
                chosen_mid_probs = mid_probs_list[offset_idx]

                if layer_alpha == 0:
                    idea_probs = probs_final - chosen_mid_probs
                else:
                    idea_probs = (1 + layer_alpha)*probs_final - layer_alpha*chosen_mid_probs

                next_token_id = torch.argmax(idea_probs).unsqueeze(0)

                del chosen_mid_probs, idea_probs, jsd_divergences, mid_probs_list

            if next_token_id.item() in self.stop_word_ids:
                del next_token_id, probs_final
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)
            del next_token_id, probs_final

            if self.device == "cuda":
                clear_cuda_cache()

        output_text = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        del generated_context, generated_no_context
        if self.device == "cuda":
            clear_cuda_cache()
        return output_text

    # --------------------------------------------------------------------------------
    # (A-B vs A-B) JSD
    # --------------------------------------------------------------------------------

    #****************************************************
    # Developed By: Syed Kamal Ahmed Hiron (IIT Patna)
    #****************************************************
    def _generate_advanced_contrast_layer_adjusted_context_jsd(
        self,
        context_ids,
        no_context_ids,
        max_new_tokens,
        initial_length_context,
        alpha,
        layer_alpha,
        start_layer
    ):
        """ 
        Key improvements:
        1. Top-2 layer ensemble with exponential JSD weighting (strong emphasis on best)
        2. VERY high adaptive layer_alpha (0.80-0.90) based on JSD magnitude
        3. Stronger temperature sharpening (0.75-0.85) for better token discrimination
        4. Better probability fusion using weighted combination
        """
        
        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            out_ctx = self.model(generated_context, output_hidden_states=True)
            out_nocxt = self.model(generated_no_context, output_hidden_states=True)

            final_logits_context = self.model.lm_head(out_ctx.hidden_states[-1][:, -1:, :])
            final_logits_no_context = self.model.lm_head(out_nocxt.hidden_states[-1][:, -1:, :])
            final_adjusted_logits = (1 + alpha) * final_logits_context - alpha * final_logits_no_context
            probs_final = F.softmax(final_adjusted_logits, dim=-1).squeeze()
            final_top_prob, _ = torch.max(probs_final, dim=-1)
            final_top_prob_val = final_top_prob.item()

            jsd_divergences = []
            mid_probs_list = []
            mid_top_probs_list = []

            for layer_idx, (ctx_hid, nocxt_hid) in enumerate(
                zip(out_ctx.hidden_states, out_nocxt.hidden_states)
            ):
                if layer_idx >= start_layer:
                    mid_logits_ctx = self.model.lm_head(ctx_hid[:, -1:, :])
                    mid_logits_nocxt = self.model.lm_head(nocxt_hid[:, -1:, :])
                    mid_adjusted_logits = (1 + alpha) * mid_logits_ctx - alpha * mid_logits_nocxt
                    mid_probs = F.softmax(mid_adjusted_logits, dim=-1).squeeze()
                    mid_top_prob, _ = torch.max(mid_probs, dim=-1)

                    jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                    jsd_val_item = jsd_val.item()
                    
                    jsd_divergences.append((jsd_val_item, layer_idx, len(mid_probs_list), mid_top_prob.item()))
                    mid_probs_list.append(mid_probs)
                    mid_top_probs_list.append(mid_top_prob.item())

                    # Free per-layer logits to reduce peak memory
                    del mid_logits_ctx, mid_logits_nocxt, mid_adjusted_logits, jsd_val

            # We no longer need full hidden states from this step
            del out_ctx, out_nocxt, final_logits_context, final_logits_no_context, final_adjusted_logits

            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final)  # Keep as scalar, will reshape for concatenation
            else:
                # Sort by JSD descending
                sorted_jsd = sorted(jsd_divergences, key=lambda x: x[0], reverse=True)
                
                # Use top-3 layers with very strong exponential weighting
                num_layers_to_use = min(3, len(sorted_jsd))
                top_layers = sorted_jsd[:num_layers_to_use]
                best_jsd = top_layers[0][0]
                
                if num_layers_to_use == 1:
                    best_idx = top_layers[0][2]
                    if best_idx < 0 or best_idx >= len(mid_probs_list):
                        best_idx = 0
                    chosen_mid_probs = mid_probs_list[best_idx]
                    mid_top_prob_val = mid_top_probs_list[best_idx]
                else:
                    # EXTREMELY strong exponential weighting: exp(jsd * 15.0)
                    # This gives almost all weight to the best layer
                    total_weight = 0.0
                    weighted_probs = torch.zeros_like(mid_probs_list[0])
                    
                    for jsd_val, layer_idx, list_idx, conf_prob in top_layers:
                        if 0 <= list_idx < len(mid_probs_list):
                            # Extremely strong exponential weighting
                            weight = torch.exp(torch.tensor(jsd_val * 15.0, device=mid_probs_list[0].device))
                            weighted_probs += weight * mid_probs_list[list_idx]
                            total_weight += weight
                    
                    if total_weight > 1e-8:
                        chosen_mid_probs = weighted_probs / total_weight
                        best_idx = top_layers[0][2]
                        if best_idx < 0 or best_idx >= len(mid_probs_list):
                            best_idx = 0
                        mid_top_prob_val = mid_top_probs_list[best_idx]
                    else:
                        best_idx = top_layers[0][2]
                        if best_idx < 0 or best_idx >= len(mid_probs_list):
                            best_idx = 0
                        chosen_mid_probs = mid_probs_list[best_idx]
                        mid_top_prob_val = mid_top_probs_list[best_idx]
                
                # EXTREMELY aggressive adaptive layer_alpha - push to maximum
                mid_is_better = (mid_top_prob_val - final_top_prob_val) >= -0.003
                
                # Use EXTREMELY high layer_alpha - push boundaries
                if best_jsd > 0.010:
                    # Very high JSD: use maximum layer_alpha
                    effective_layer_alpha = 0.95 if mid_is_better else 0.90
                elif best_jsd > 0.007:
                    # High JSD: use very high layer_alpha
                    effective_layer_alpha = 0.90 if mid_is_better else 0.85
                elif best_jsd > 0.004:
                    # Medium JSD: use high layer_alpha
                    effective_layer_alpha = 0.85 if mid_is_better else 0.80
                else:
                    # Low JSD: still use very high layer_alpha
                    effective_layer_alpha = 0.80
                
                # HYBRID combination: Use both standard LACD and a more conservative fusion,
                # then fall back to the baseline distribution when the advanced head looks unreliable.
                # Standard LACD-style combination
                idea_logits_standard = (1 + effective_layer_alpha) * torch.log(probs_final + 1e-10) \
                                       - effective_layer_alpha * torch.log(chosen_mid_probs + 1e-10)
                idea_probs_standard = F.softmax(idea_logits_standard, dim=-1)

                # Conservative fusion: simple average of logits (less aggressive than standard)
                conservative_logits = 0.5 * torch.log(probs_final + 1e-10) + 0.5 * torch.log(chosen_mid_probs + 1e-10)
                idea_probs_conservative = F.softmax(conservative_logits, dim=-1)

                # Blend aggressive and conservative heads
                idea_probs = 0.7 * idea_probs_standard + 0.3 * idea_probs_conservative
                idea_probs = idea_probs / (idea_probs.sum() + 1e-10)

                # Temperature sharpening, but not excessively low (to avoid over-confident wrong tokens)
                prob_entropy = -torch.sum(idea_probs * torch.log(idea_probs + 1e-10))
                entropy_norm = prob_entropy.item() / 12.0

                if entropy_norm > 0.6:
                    temp = 0.8   # mild sharpening in high uncertainty
                elif entropy_norm > 0.4:
                    temp = 0.75  # moderate sharpening
                else:
                    temp = 0.7   # stronger when already confident

                sharpened_logits = torch.log(idea_probs + 1e-10) / temp
                idea_probs = F.softmax(sharpened_logits, dim=-1)

                # Compare with baseline LACD distribution; if advanced head is less confident,
                # fall back to baseline token choice. This tends to improve Exact Match in practice.
                base_top_prob, base_top_idx = torch.max(probs_final, dim=-1)
                adv_top_prob, adv_top_idx = torch.max(idea_probs, dim=-1)

                if adv_top_prob < base_top_prob * 0.9:
                    # Advanced is noticeably less confident -> use baseline token
                    next_token_id = base_top_idx  # Keep as scalar, will reshape for concatenation
                else:
                    next_token_id = adv_top_idx  # Keep as scalar, will reshape for concatenation

                del idea_probs, idea_probs_conservative, idea_probs_standard
                del chosen_mid_probs, mid_probs_list, mid_top_probs_list, jsd_divergences
                # Note: probs_final is deleted later on line 819 (or 814 if breaking early)

            if next_token_id.item() in self.stop_word_ids:
                del next_token_id, probs_final
                break

            # Reshape scalar to [1, 1] for concatenation with [1, seq_len]
            next_token_id_2d = next_token_id.unsqueeze(0).unsqueeze(-1)
            generated_context = torch.cat([generated_context, next_token_id_2d], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id_2d], dim=-1)
            del next_token_id, next_token_id_2d, probs_final

            if self.device == "cuda":
                clear_cuda_cache()

        output_text = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        del generated_context, generated_no_context
        if self.device == "cuda":
            clear_cuda_cache()
        return output_text

    def _generate_contrast_layer_adjusted_context_jsd_subset(
        self,
        context_ids,
        no_context_ids,
        max_new_tokens,
        initial_length_context,
        alpha,
        layer_alpha,
        subset_layers
    ):
        """
        Similar to the above, but only checks the specified subset of layers instead of all layers >= start_layer.
        """
        if subset_layers is None:
            subset_layers = []

        generated_context = context_ids.clone()
        generated_no_context = no_context_ids.clone()

        for _ in range(max_new_tokens):
            logits_ctx, hidden_states_ctx = self.adapter.forward_with_hidden_states(generated_context)
            logits_nocxt, hidden_states_nocxt = self.adapter.forward_with_hidden_states(generated_no_context)

            final_logits_ctx = 0
            final_logits_nocxt = 0
            # Final distribution (A-B)
            if hidden_states_ctx and hidden_states_nocxt:
                final_logits_ctx = self.adapter.get_logits_from_hidden_state(hidden_states_ctx[-1][:, -1:, :])
                final_logits_nocxt = self.adapter.get_logits_from_hidden_state(hidden_states_nocxt[-1][:, -1:, :])
            else:
                final_logits_ctx = logits_ctx[:, -1:, :]
                final_logits_nocxt = logits_nocxt[:, -1:, :]
            final_adjusted_logits = (1 + alpha)*final_logits_ctx - alpha*final_logits_nocxt
            probs_final = F.softmax(final_adjusted_logits, dim=-1).squeeze()

            jsd_divergences = []
            mid_probs_list = []

            if hidden_states_ctx and hidden_states_nocxt:
                for layer_idx in subset_layers:
                    if 0 <= layer_idx < len(hidden_states_ctx):
                        mid_logits_ctx = self.adapter.get_logits_from_hidden_state(hidden_states_ctx[layer_idx][:, -1:, :])
                        mid_logits_nocxt = self.adapter.get_logits_from_hidden_state(hidden_states_nocxt[layer_idx][:, -1:, :])
                        mid_adjusted_logits = (1 + alpha)*mid_logits_ctx - alpha*mid_logits_nocxt
                        mid_probs = F.softmax(mid_adjusted_logits, dim=-1).squeeze()

                        jsd_val = jensen_shannon_divergence(probs_final, mid_probs)
                        jsd_divergences.append((jsd_val.item(), layer_idx))
                        mid_probs_list.append(mid_probs)
                        
                        # Delete intermediate tensors
                        del mid_logits_ctx, mid_logits_nocxt, mid_adjusted_logits, mid_probs, jsd_val
                
                # Delete hidden states after processing
                del hidden_states_ctx, hidden_states_nocxt
            del logits_ctx, logits_nocxt

            # If subset_layers is empty or invalid, fallback
            if len(jsd_divergences) == 0:
                next_token_id = torch.argmax(probs_final).unsqueeze(0)
                del final_logits_ctx, final_logits_nocxt, final_adjusted_logits, probs_final
            else:
                _, max_jsd_idx = max(jsd_divergences, key=lambda x: x[0])

                # find the correct offset in mid_probs_list
                chosen_mid_probs = None
                for i, (val, lidx) in enumerate(jsd_divergences):
                    if lidx == max_jsd_idx:
                        chosen_mid_probs = mid_probs_list[i]
                        break

                if chosen_mid_probs is None:
                    next_token_id = torch.argmax(probs_final).unsqueeze(0)
                    del final_logits_ctx, final_logits_nocxt, final_adjusted_logits, probs_final, mid_probs_list, jsd_divergences
                else:
                    if layer_alpha == 0:
                        idea_probs = probs_final - chosen_mid_probs
                    else:
                        idea_probs = (1 + layer_alpha)*probs_final - layer_alpha * chosen_mid_probs

                    # Optionally re-softmax
                    idea_probs = F.softmax(idea_probs, dim=-1)
                    next_token_id = torch.argmax(idea_probs).unsqueeze(0)
                    
                    # Delete intermediate tensors
                    del final_logits_ctx, final_logits_nocxt, final_adjusted_logits, probs_final
                    del chosen_mid_probs, idea_probs, mid_probs_list, jsd_divergences

            if next_token_id.item() in self.stop_word_ids:
                del next_token_id
                break

            generated_context = torch.cat([generated_context, next_token_id.unsqueeze(0)], dim=-1)
            generated_no_context = torch.cat([generated_no_context, next_token_id.unsqueeze(0)], dim=-1)
            del next_token_id
            
            # Clear cache after each token
            if self.device == "cuda":
                clear_cuda_cache()

        output_text = self.tokenizer.decode(
            generated_context[0, initial_length_context:], skip_special_tokens=True
        )
        # Delete generated tensors
        del generated_context, generated_no_context
        # Clear cache after generation
        if self.device == "cuda":
            clear_cuda_cache()
        return output_text
