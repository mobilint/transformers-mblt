"""Qwen3-ASR Transformers backend, vendored from ``qwen-asr`` 0.0.6 (``qwen_asr/core/transformers_backend``).

Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. Licensed under the Apache License,
Version 2.0 (see the header of each module). Vendored so that Qwen3-ASR does not need the ``qwen-asr``
distribution, which pins ``transformers==4.57.6``; the modules are modified to import under Transformers 5 as
well. Mobilint's NPU classes subclass these in the parent package.
"""
