# NOTE: Florence-2's modeling and configuration modules (loaded via
# trust_remote_code) predate several transformers >= 4.49 API contracts. The
# shims below restore the older defaults so the legacy trust_remote_code
# modules keep working without forking the upstream Florence-2 repos.
from transformers import AutoModelForCausalLM, AutoProcessor
from collections import OrderedDict

from optimum.quanto import freeze
from toolkit.basic import flush
from toolkit.util.quantize import quantize, get_qtype

from .BaseCaptioner import BaseCaptioner
import transformers
import logging
import warnings

from transformers import PretrainedConfig, PreTrainedModel

# Compatibility shim: Florence-2's custom configuration_florence2.py
# accesses forced_bos_token_id / forced_eos_token_id, which were
# removed from PretrainedConfig in transformers >= 4.49 (moved to
# GenerationConfig). Restore them as class-level defaults so legacy
# trust_remote_code configs keep working.
if not hasattr(PretrainedConfig, "forced_bos_token_id"):
    PretrainedConfig.forced_bos_token_id = None
if not hasattr(PretrainedConfig, "forced_eos_token_id"):
    PretrainedConfig.forced_eos_token_id = None

# Compatibility shim: transformers 4.49+ rewrote
# get_expanded_tied_weights_keys to assume _tied_weights_keys is a
# dict[str, str] (mapping target -> source) and unconditionally calls
# .keys() / .values() on it. Florence-2's modeling_florence2.py
# predates that contract and declares the older list[str] format.
# When we see a list, return it unchanged -- the pre-4.49 API already
# gave the fully-expanded key set with no per-key expansion needed.
# Method patching (vs. data-attribute setattr) is safe here because
# Python's bound-method lookup walks the class MRO normally and is
# not routed through nn.Module.__getattr__.
_orig_get_expanded_tied_weights_keys = getattr(
    PreTrainedModel, "get_expanded_tied_weights_keys", None
)
if _orig_get_expanded_tied_weights_keys is not None:
    def _get_expanded_tied_weights_keys_compat(self):
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, list):
            return list(tied)
        return _orig_get_expanded_tied_weights_keys(self)

    PreTrainedModel.get_expanded_tied_weights_keys = (
        _get_expanded_tied_weights_keys_compat
    )

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


# Task token compatibility by model family. Unknown prompts fall back to
# <MORE_DETAILED_CAPTION>; using a token a model wasn't trained on usually
# yields a degenerate or empty caption.
#   microsoft/Florence-2-*, thwri/CogFlorence-2.*-Large
#       -> <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>
#   HuggingFaceM4/Florence-2-DocVQA
#       -> <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>
#   MiaoshouAI/Florence-2-*-PromptGen-v1.5
#       -> the three above + <GENERATE_TAGS>, <MIXED_CAPTION>
#   MiaoshouAI/Florence-2-*-PromptGen-v2.0
#       -> the v1.5 set + <ANALYZE>, <MIXED_CAPTION_PLUS>
#   gokaygokay/Florence-2-Flux-Large, gokaygokay/Florence-2-SD3-Captioner
#       -> <DESCRIPTION>
#   PJMixers-Images/Florence-2-base-Castollux-v0.5
#       -> <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>
VALID_FLORENCE2_TASKS = (
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<GENERATE_TAGS>",
    "<MIXED_CAPTION>",
    "<MIXED_CAPTION_PLUS>",
    "<ANALYZE>",
    "<DESCRIPTION>",
)


class Florence2Captioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Florence2Captioner, self).__init__(process_id, job, config, **kwargs)

    def load_model(self):
        self.print_and_status_update("Loading Florence-2 model")
        # attn_implementation="eager" forces transformers to skip the
        # _check_and_adjust_attn_implementation auto-detection path, which in
        # 4.49+ reads class-level _supports_sdpa / _supports_flash_attn_2
        # flags that Florence-2's modeling_florence2.py never declared.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="cpu",
        )
        if not self.caption_config.low_vram:
            self.model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing Florence-2 model")
            try:
                quantize(self.model, weights=get_qtype(self.caption_config.qtype))
                freeze(self.model)
                flush()
            except Exception as e:
                # Florence-2 ships custom modules via trust_remote_code that
                # quanto cannot always introspect; fall back to unquantized.
                print(f"Florence-2 quantization skipped: {e}")
                flush()
        self.processor = AutoProcessor.from_pretrained(
            self.caption_config.model_name_or_path,
            trust_remote_code=True,
        )
        if self.caption_config.low_vram:
            self.model.to(self.device_torch)
        flush()

    def _resolve_task(self) -> str:
        prompt = (self.caption_config.caption_prompt or "").strip()
        if prompt in VALID_FLORENCE2_TASKS:
            return prompt
        return "<MORE_DETAILED_CAPTION>"

    def get_caption_for_file(self, file_path: str) -> str:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        try:
            task = self._resolve_task()
            inputs = self.processor(
                text=task,
                images=img,
                return_tensors="pt",
            ).to(self.device_torch, self.torch_dtype)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.caption_config.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(img.width, img.height),
            )

            caption = parsed.get(task, "")
            if not isinstance(caption, str):
                caption = str(caption)
            return caption.strip()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
