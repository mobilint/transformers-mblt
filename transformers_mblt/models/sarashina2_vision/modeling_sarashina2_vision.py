"""Embedding-only host model for the static Sarashina2.2-Vision MXQ pair."""
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.generation_utils import MobilintGenerationMixin
from ..llama.modeling_llama import MobilintLlamaForCausalLM
from .configuration_sarashina2_vision import MobilintSarashina2VisionConfig


class _SarashinaDecoder(MobilintLlamaForCausalLM):
    """Use the shared cache/chunk/logit machinery with Sarashina's infer ABI."""

    def llm_forward(self, inputs_embeds, past_key_values, cache_position,
                    npu_prefill_chunk_size=None, count_npu_time=False,
                    attention_mask=None, logits_to_keep=1, **kwargs):
        chunk_size = self.resolve_npu_prefill_chunk_size(npu_prefill_chunk_size)
        if not 1 <= chunk_size <= 128:
            raise ValueError("Sarashina npu_prefill_chunk_size must be between 1 and 128.")
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if seen + inputs_embeds.shape[1] > 4096:
            raise ValueError("Sarashina's compiled KV cache holds at most 4096 tokens.")
        self._bind_single_slot_cache(past_key_values)
        x = inputs_embeds.detach().float().cpu().numpy()[:, None, :, :]
        self.npu_time = 0.0 if count_npu_time else None

        def infer(start, end):
            cache_size = seen + start
            t0 = time.perf_counter() if count_npu_time else None
            result = self.npu_backend.mxq_models[0].infer(
                [np.ascontiguousarray(x[:, :, start:end, :])], cache_size=cache_size)
            if count_npu_time:
                elapsed = time.perf_counter() - t0
                self.npu_time += elapsed
                self._record_npu_timing("prefill" if seen == 0 else "decode", elapsed)
            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start:end])
            return np.asarray(result[0]).reshape(1, 1, -1, self.config.vocab_size)

        logits = self._run_chunked_logits_to_keep(
            do_infer=infer, seq_len=x.shape[2], npu_prefill_chunk_size=chunk_size,
            logits_to_keep=logits_to_keep, dtype=torch.float32, device=inputs_embeds.device)
        return logits.squeeze(0)


class MobilintSarashina2VisionForCausalLM(PretrainedOnlyMixin, MobilintGenerationMixin):
    config_class = MobilintSarashina2VisionConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")

    def __init__(self, config, *args, **kwargs):
        no_launch = kwargs.pop("no_launch", False)
        super().__init__(config, *args, **kwargs)
        if config.text_config.max_batch_size != 1 or config.vision_config.max_batch_size != 1:
            raise ValueError("The static Sarashina release supports batch size 1 only.")
        self.model = _SarashinaDecoder(config.text_config, _internal_call=True, no_launch=True)
        self.vision_backend = config.vision_config.npu_backend
        self.vision_backend.name_or_path = config.name_or_path
        self.vision_backend.revision = getattr(config, "revision", None)
        self.vision_backend._commit_hash = getattr(config, "_commit_hash", None)
        try:
            self.vision_backend.create()
            if not no_launch:
                self.launch()
        except Exception:
            self.dispose()
            raise

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_cache_mxq_model(self):
        return self.model.get_cache_mxq_model()

    def _get_cache(self, *args, **kwargs):
        return self.model._get_cache(*args, **kwargs)

    def launch(self):
        self.model.launch()
        self.vision_backend.launch()

    def dispose(self):
        self.model.dispose()
        self.vision_backend.dispose()

    def get_image_embeds(self, pixel_values, image_grid_thw):
        if image_grid_thw is None or image_grid_thw.tolist() != [[1, 32, 32]]:
            raise ValueError("Static Sarashina requires exactly one image with image_grid_thw=[[1,32,32]].")
        if tuple(pixel_values.shape) != (1024, 1176):
            raise ValueError("Static Sarashina requires pixel_values shaped (1024,1176).")
        patches = np.ascontiguousarray(pixel_values.detach().float().cpu().numpy().reshape(1, 1024, 1176))
        output = self.vision_backend.mxq_models[0].infer([patches])[0]
        features = torch.from_numpy(np.asarray(output, dtype=np.float32).copy())
        expected = 256 * self.config.text_config.hidden_size
        if features.numel() != expected:
            raise ValueError(f"Vision MXQ must produce 256 image tokens; got output shape {features.shape}.")
        return features.reshape(256, self.config.text_config.hidden_size)

    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None, attention_mask=None,
                position_ids=None, past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=1,
                npu_prefill_chunk_size=None, count_npu_time=False, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if use_cache is False:
            raise ValueError("Sarashina NPU inference requires use_cache=True.")
        if input_ids is not None and input_ids.shape[0] != 1:
            raise ValueError("Static Sarashina supports batch size 1 only.")
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise ValueError("Static Sarashina does not support padded attention masks.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("Static Sarashina supports batch size 1 only.")
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("Image insertion requires input_ids to locate image placeholders.")
            features = self.get_image_embeds(pixel_values, image_grid_thw).to(inputs_embeds)
            mask = input_ids == self.config.image_token_index
            if int(mask.sum()) != 256:
                raise ValueError("The image requires exactly 256 image placeholder tokens.")
            inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1), features)
        # Packaged token embeddings and the compiled vision head share the
        # decoder's rotation basis; neither needs a host-side transformation.
        return self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, labels=labels, use_cache=True,
            cache_position=cache_position, logits_to_keep=logits_to_keep,
            npu_prefill_chunk_size=npu_prefill_chunk_size, count_npu_time=count_npu_time, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, pixel_values=None,
                                      image_grid_thw=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs)
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            model_inputs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        else:
            model_inputs.pop("pixel_values", None)
            model_inputs.pop("image_grid_thw", None)
        return model_inputs

    def generate(self, *args, **kwargs):
        config = kwargs.get("generation_config", self.generation_config)
        if kwargs.get("use_cache", getattr(config, "use_cache", True)) is False:
            raise ValueError("Sarashina NPU generation requires use_cache=True.")
        if (kwargs.get("num_beams", getattr(config, "num_beams", 1)) or 1) != 1:
            raise ValueError("Static Sarashina supports greedy/sampling generation, not beam search.")
        kwargs["use_cache"] = True
        return super().generate(*args, **kwargs)


AutoModel.register(MobilintSarashina2VisionConfig, MobilintSarashina2VisionForCausalLM)
AutoModelForCausalLM.register(MobilintSarashina2VisionConfig, MobilintSarashina2VisionForCausalLM)
AutoModelForImageTextToText.register(MobilintSarashina2VisionConfig, MobilintSarashina2VisionForCausalLM)
