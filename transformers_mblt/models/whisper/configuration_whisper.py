from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.models.whisper.configuration_whisper import WhisperConfig

from ...utils.configuration_utils import MobilintEncoderDecoderConfigMixin


class MobilintWhisperConfig(MobilintEncoderDecoderConfigMixin, WhisperConfig):
    model_type = "mobilint-whisper"

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        
        self.tie_word_embeddings = False

AutoConfig.register("mobilint-whisper", MobilintWhisperConfig)
# transformers 5.0-5.2 load a model type missing from this table with the generic TokenizersBackend, ignoring the
# Hub's tokenizer_class, and the ASR pipeline then fails on WhisperTokenizer._decode_asr. 5.3 fixed it upstream.
TOKENIZER_MAPPING_NAMES.setdefault("mobilint-whisper", TOKENIZER_MAPPING_NAMES["whisper"])
