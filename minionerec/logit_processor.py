from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        self.max_warning_per_step = 3
        self.max_total_warnings = 40
        self.no_valid_total = 0
        self.backoff_hit_total = 0
        self._step_warning_count = {}
        self._total_warning_count = 0
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    def _get_allowed_tokens_with_backoff(self, batch_id: int, hash_key: List[int]) -> List[int]:
        allowed = self._prefix_allowed_tokens_fn(batch_id, hash_key)
        if allowed:
            return allowed

        for start in range(1, len(hash_key)):
            fallback_key = hash_key[start:]
            allowed = self._prefix_allowed_tokens_fn(batch_id, fallback_key)
            if allowed:
                self.backoff_hit_total += 1
                return allowed
        return []

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._get_allowed_tokens_with_backoff(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    self.no_valid_total += 1
                    step_warn_count = self._step_warning_count.get(self.count, 0) + 1
                    self._step_warning_count[self.count] = step_warn_count
                    if self._total_warning_count < self.max_total_warnings and step_warn_count <= self.max_warning_per_step:
                        warnings.warn(
                            f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                            f"Falling back to EOS for this beam. "
                        )
                        self._total_warning_count += 1
                    elif self._total_warning_count < self.max_total_warnings and step_warn_count == self.max_warning_per_step + 1:
                        warnings.warn(
                            f"Step {self.count}: too many no-valid-token events, suppressing further warnings. "
                            f"total_no_valid={self.no_valid_total}, backoff_hit={self.backoff_hit_total}"
                        )
                        self._total_warning_count += 1
                    elif self._total_warning_count == self.max_total_warnings:
                        warnings.warn(
                            f"No-valid-token warnings reached global cap ({self.max_total_warnings}), "
                            f"further warnings are suppressed. total_no_valid={self.no_valid_total}, "
                            f"backoff_hit={self.backoff_hit_total}"
                        )
                        self._total_warning_count += 1
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores