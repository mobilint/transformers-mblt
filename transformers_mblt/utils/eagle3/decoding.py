"""EAGLE-3 decoding helpers.

This module is a stable import surface for tree-decoding primitives.
"""

from .tree_decoding import (
    Eagle3DecodeStep,
    decode_step,
    evaluate_posterior,
    initialize_tree,
    prepare_logits_processor,
    tree_decoding,
    update_inference_inputs,
)

__all__ = [
    "Eagle3DecodeStep",
    "decode_step",
    "prepare_logits_processor",
    "initialize_tree",
    "tree_decoding",
    "evaluate_posterior",
    "update_inference_inputs",
]
