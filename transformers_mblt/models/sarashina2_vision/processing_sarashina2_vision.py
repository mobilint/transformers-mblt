"""Static BICUBIC 448x448 preprocessing for Sarashina2.2-Vision."""
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import to_pil_image
from transformers.image_utils import make_flat_list_of_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin

from .configuration_sarashina2_vision import MobilintSarashina2VisionConfig


class _SarashinaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}


class MobilintSarashina2VisionProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template"]

    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        self.image_token = "<|file|>"
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load only standard processor/tokenizer resources, never upstream remote
        # model code or GPU weights. This also accepts the upstream image config.
        kwargs.pop("trust_remote_code", None)
        image_processor = Qwen2VLImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False, **kwargs)
        processor_dict, _ = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor, tokenizer, chat_template=processor_dict.get("chat_template", tokenizer.chat_template))

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if videos is not None:
            raise NotImplementedError("The static Sarashina release does not support video.")
        texts = [text] if isinstance(text, str) else list(text or [])
        if len(texts) != 1:
            raise ValueError("The static Sarashina release requires exactly one text prompt.")
        output_kwargs = self._merge_kwargs(_SarashinaProcessorKwargs,
                                           tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)
        text_kwargs = output_kwargs["text_kwargs"]
        return_tensors = text_kwargs.pop("return_tensors", None)
        image_inputs = {}
        placeholders = texts[0].count(self.image_token)
        if images is not None:
            images = make_flat_list_of_images(images)
            if len(images) != 1 or placeholders != 1:
                raise ValueError("Static Sarashina requires one image and one <|file|> placeholder per prompt.")
            image = images[0]
            if not isinstance(image, Image.Image):
                image = to_pil_image(image)
            image = image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
            image_kwargs = dict(output_kwargs["images_kwargs"])
            # Resize before rescale/normalize, precisely as the verified runner.
            image_kwargs["do_resize"] = False
            image_inputs = self.image_processor(images=[image], **image_kwargs)
            grid = image_inputs["image_grid_thw"]
            if list(grid[0]) != [1, 32, 32]:
                raise ValueError("Image processor must produce the static [1,32,32] grid.")
            texts[0] = texts[0].replace(self.image_token, self.image_token * 256)
        elif placeholders:
            raise ValueError("An image placeholder was supplied without an image.")
        text_inputs = self.tokenizer(texts, **text_kwargs)
        # Truncation must not silently drop a portion of an image's token block.
        if images is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            ids = text_inputs["input_ids"][0]
            if sum(int(token) == image_token_id for token in ids) != 256:
                raise ValueError("Tokenization/truncation must preserve all 256 image tokens.")
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs, **kwargs):
        return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + ["pixel_values", "image_grid_thw"]))


AutoProcessor.register(MobilintSarashina2VisionConfig, MobilintSarashina2VisionProcessor)
