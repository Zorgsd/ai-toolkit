# NOTE: Florence-2's modeling and configuration modules (loaded via
# trust_remote_code) predate several transformers >= 4.49 API contracts. The
# shims below restore the older defaults so the legacy trust_remote_code
# modules keep working without forking the upstream Florence-2 repos.
from transformers import AutoModelForCausalLM, AutoProcessor
from collections import OrderedDict

import torch
from optimum.quanto import freeze
from toolkit.basic import flush
from toolkit.util.quantize import quantize, get_qtype

from .BaseCaptioner import BaseCaptioner
import transformers
import logging
import warnings

from transformers import PretrainedConfig, PreTrainedModel, RobertaTokenizer

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
# dict[str, str] mapping ABSOLUTE paths from the model root
# (target -> source). The contract is enforced in two places:
#   1. post_init stores the result on self.all_tied_weights_keys and
#      calls .update() on it (return value must be a dict, not a list).
#   2. _finalize_model_loading -> mark_tied_weights_as_initialized
#      walks the dict and calls self.get_parameter(tied_param) on each
#      value, expecting an absolute submodule path.
# Florence-2's modeling_florence2.py declares the older list[str]
# format whose entries (e.g. "encoder.embed_tokens.weight") were
# implicitly resolved relative to the *base model's* submodule path,
# not the outer ConditionalGeneration wrapper. There's no general way
# to recover the correct absolute prefix for an arbitrary Florence-2
# submodel from the legacy list, so any concrete dict we synthesize
# (including the earlier self-tie {k: k for k in tied}) will route
# get_parameter() through a non-existent attribute and crash.
# Returning {} makes transformers' post-load tied-weight VALIDATION
# pass a no-op for these models. The actual weight tying still works
# -- Florence-2's modeling code calls tie_weights() via the standard
# PyTorch mechanism, which reads _tied_weights_keys directly and is
# unaffected by what this method returns. We're only disabling the
# 4.49+ post-load initialization audit, not the tying itself.
# *args/**kwargs are dropped on the list branch on purpose (the
# pre-4.49 API took no parameters) and forwarded on the dict branch
# so parameters added by future transformers versions (e.g. the
# current all_submodels=False) keep flowing through.
# Method patching (vs. data-attribute setattr) is safe here because
# Python's bound-method lookup walks the class MRO normally and is
# not routed through nn.Module.__getattr__.
_orig_get_expanded_tied_weights_keys = getattr(
    PreTrainedModel, "get_expanded_tied_weights_keys", None
)
if _orig_get_expanded_tied_weights_keys is not None:
    def _get_expanded_tied_weights_keys_compat(self, *args, **kwargs):
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, list):
            # Legacy list[str] entries are relative to the base model's
            # submodule path, which we can't reliably recover here for
            # an arbitrary Florence-2 submodel. Return {} so the new
            # tied-weight validation no-ops; tying still happens via
            # the standard PyTorch path that reads _tied_weights_keys
            # directly.
            return {}
        return _orig_get_expanded_tied_weights_keys(self, *args, **kwargs)

    PreTrainedModel.get_expanded_tied_weights_keys = (
        _get_expanded_tied_weights_keys_compat
    )

# Compatibility shim: transformers >= 4.49 dropped the public
# additional_special_tokens property from the slow tokenizer path,
# leaving it only on the *Fast variants. Florence-2's
# processing_florence2.py reads tokenizer.additional_special_tokens
# unconditionally during processor construction, and AutoProcessor's
# trust_remote_code path silently drops use_fast=True in its kwargs
# filtering, so we can't reach the Fast tokenizer through the normal
# from_pretrained call. Restore the property on the slow class only
# (RobertaTokenizer, not its parent PreTrainedTokenizer): scoping it
# narrowly avoids silently changing behavior for every other slow
# tokenizer in the project (T5, CLIP, BERT, ...).
# Property patching (vs. data-attribute setattr on a model) is safe
# for the same reason method patching is: Python's descriptor protocol
# resolves class-level descriptors via the MRO and is not routed
# through nn.Module.__getattr__ -- and PreTrainedTokenizer is a plain
# object anyway, so __getattr__ semantics aren't even in play.
if not hasattr(RobertaTokenizer, "additional_special_tokens"):
    def _additional_special_tokens_compat_get(self):
        # Prefer the internal that older code still populates during
        # init; fall back to the special_tokens_map view, which both
        # slow and fast tokenizers continue to expose.
        val = getattr(self, "_additional_special_tokens", None)
        if val is not None:
            return list(val)
        return list(
            self.special_tokens_map.get("additional_special_tokens", [])
        )

    def _additional_special_tokens_compat_set(self, value):
        self._additional_special_tokens = (
            list(value) if value is not None else []
        )

    RobertaTokenizer.additional_special_tokens = property(
        _additional_special_tokens_compat_get,
        _additional_special_tokens_compat_set,
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
        # Florence-2 was trained in fp16; bf16 (the ai-toolkit default) degrades
        # the visual encoder enough to produce nonsense captions. ComfyUI's
        # Florence-2 nodes use precision="fp16" for the same reason.
        florence2_dtype = torch.float16
        self._florence2_dtype = florence2_dtype
        print(
            f"[Florence2 DEBUG] forcing dtype to {florence2_dtype} "
            f"(was {self.torch_dtype})",
            flush=True,
        )
        # attn_implementation="eager" forces transformers to skip the
        # _check_and_adjust_attn_implementation auto-detection path, which in
        # 4.49+ reads class-level _supports_sdpa / _supports_flash_attn_2
        # flags that Florence-2's modeling_florence2.py never declared.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=florence2_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="cpu",
        )
        # Compatibility shim: in transformers >= 4.49 GenerationMixin is no
        # longer auto-inherited by base model classes -- models must declare
        # it explicitly. Florence-2's modeling_florence2.py predates that
        # contract, so the inner Florence2LanguageForConditionalGeneration
        # has no .generate() and self.model.generate() crashes when it
        # delegates inward. Mix GenerationMixin into the inner LM class's
        # MRO at runtime. We do this lazily here (not in the module-level
        # shim block) because the class only exists once trust_remote_code
        # has executed modeling_florence2.py during from_pretrained.
        # Inject as a base class instead of copying methods so the rest of
        # GenerationMixin's machinery (cache classes, beam search hooks,
        # etc.) stays internally consistent.
        from transformers import GenerationMixin
        inner_lm_cls = type(self.model.language_model)
        print(
            f"[Florence2 DEBUG] outer model class: {type(self.model).__name__}",
            flush=True,
        )
        print(
            f"[Florence2 DEBUG] inner_lm_cls: {inner_lm_cls.__name__}",
            flush=True,
        )
        print(
            f"[Florence2 DEBUG] before patch -- hasattr(inner_lm_cls, 'generate') = "
            f"{hasattr(inner_lm_cls, 'generate')}",
            flush=True,
        )
        print(
            f"[Florence2 DEBUG] before patch -- inner_lm_cls.__mro__ = "
            f"{str(inner_lm_cls.__mro__)}",
            flush=True,
        )
        if not hasattr(inner_lm_cls, "generate"):
            inner_lm_cls.__bases__ = (GenerationMixin,) + inner_lm_cls.__bases__
        print(
            f"[Florence2 DEBUG] after patch  -- hasattr(inner_lm_cls, 'generate') = "
            f"{hasattr(inner_lm_cls, 'generate')}",
            flush=True,
        )
        print(
            f"[Florence2 DEBUG] after patch  -- inner_lm_cls.__mro__ = "
            f"{str(inner_lm_cls.__mro__)}",
            flush=True,
        )
        # Florence-2's outer wrapper has generation_config (loaded from
        # generation_config.json), but the inner language model doesn't --
        # transformers >= 4.49's GenerationMixin.generate() reads
        # self.generation_config and crashes if it's missing. Copy the outer
        # config down so the delegated generate() call can find it.
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            self.model.language_model.generation_config = self.model.generation_config
        else:
            # Fallback: build a default GenerationConfig if the outer model
            # doesn't have one either (shouldn't happen for Florence-2 but
            # belt-and-braces).
            from transformers import GenerationConfig
            self.model.language_model.generation_config = GenerationConfig()
        print(
            f"[Florence2 DEBUG] inner generation_config set: "
            f"{self.model.language_model.generation_config is not None}",
            flush=True,
        )
        # Florence-2 (BART-based encoder-decoder) needs decoder_start_token_id,
        # bos_token_id, eos_token_id, pad_token_id on generation_config.
        # transformers >=4.49 GenerationMixin no longer falls back to reading
        # them from model.config -- they must be set explicitly.
        text_cfg = getattr(self.model.config, 'text_config', None)
        inner_gen_cfg = self.model.language_model.generation_config

        # Florence-2 BART defaults from its tokenizer/config
        _FLORENCE2_BART_DEFAULTS = {
            'decoder_start_token_id': 2,  # </s>
            'bos_token_id': 0,             # <s>
            'eos_token_id': 2,             # </s>
            'pad_token_id': 1,             # <pad>
        }

        for token_attr, default_val in _FLORENCE2_BART_DEFAULTS.items():
            val = None
            if text_cfg is not None:
                val = getattr(text_cfg, token_attr, None)
            if val is None:
                val = getattr(self.model.config, token_attr, None)
            if val is None:
                val = default_val
            setattr(inner_gen_cfg, token_attr, val)

        print(
            f"[Florence2 DEBUG] tokens on inner_gen_cfg: "
            f"decoder_start={inner_gen_cfg.decoder_start_token_id}, "
            f"bos={inner_gen_cfg.bos_token_id}, "
            f"eos={inner_gen_cfg.eos_token_id}, "
            f"pad={inner_gen_cfg.pad_token_id}",
            flush=True,
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
        # Florence-2's processor tokenizer doesn't auto-register task tokens
        # as special, so BPE splits e.g. "<MORE_DETAILED_CAPTION>" into 7
        # fragment tokens. Explicitly add them as special tokens so they
        # tokenize as a single atomic id -- matches what ComfyUI's
        # Florence2Run node does internally.
        FLORENCE2_TASK_TOKENS = [
            '<CAPTION>',
            '<DETAILED_CAPTION>',
            '<MORE_DETAILED_CAPTION>',
            '<OD>',
            '<DENSE_REGION_CAPTION>',
            '<REGION_PROPOSAL>',
            '<CAPTION_TO_PHRASE_GROUNDING>',
            '<REFERRING_EXPRESSION_SEGMENTATION>',
            '<REGION_TO_SEGMENTATION>',
            '<OPEN_VOCABULARY_DETECTION>',
            '<REGION_TO_CATEGORY>',
            '<REGION_TO_DESCRIPTION>',
            '<OCR>',
            '<OCR_WITH_REGION>',
            # MiaoshouAI PromptGen-specific tokens
            '<GENERATE_TAGS>',
            '<MIXED_CAPTION>',
            '<MIXED_CAPTION_PLUS>',
            '<ANALYZE>',
            # gokaygokay-specific
            '<DESCRIPTION>',
        ]
        num_added = self.processor.tokenizer.add_tokens(
            FLORENCE2_TASK_TOKENS,
            special_tokens=True,
        )
        print(
            f"[Florence2 DEBUG] added {num_added} task tokens as special; "
            f"sample: {self.processor.tokenizer.tokenize('<MORE_DETAILED_CAPTION>')}",
            flush=True,
        )
        if self.caption_config.low_vram:
            self.model.to(self.device_torch)
        flush()

    def _resolve_task(self) -> str:
        prompt = (self.caption_config.caption_prompt or "").strip()
        if prompt in VALID_FLORENCE2_TASKS:
            return prompt
        return "<MORE_DETAILED_CAPTION>"

    def _pad_to_square(self, img):
        from PIL import Image
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        # White pad -- Florence-2's vision encoder is sensitive to black
        # borders (it was trained on natural images). White matches what
        # ComfyUI's Florence-2 nodes do and produces coherent captions.
        padded = Image.new(img.mode, (side, side), color=(255, 255, 255))
        paste_x = (side - w) // 2
        paste_y = (side - h) // 2
        padded.paste(img, (paste_x, paste_y))
        return padded

    def get_caption_for_file(self, file_path: str) -> str:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        img = self._pad_to_square(img)
        try:
            task = self._resolve_task()
            inputs = self.processor(
                text=task,
                images=img,
                return_tensors="pt",
            ).to(self.device_torch, self._florence2_dtype)

            input_ids = inputs["input_ids"]
            print(
                f"[Florence2 DEBUG] task='{task}', input_ids shape={input_ids.shape}, "
                f"first 10 tokens={input_ids[0][:10].tolist()}",
                flush=True,
            )
            print(
                f"[Florence2 DEBUG] decoded input: "
                f"'{self.processor.tokenizer.decode(input_ids[0])}'",
                flush=True,
            )

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.caption_config.max_new_tokens,
                num_beams=3,
                do_sample=False,
                use_cache=False,  # Florence-2's decoder forward predates the
                                  # EncoderDecoderCache class API; without
                                  # disabling cache, transformers >= 4.49
                                  # raises "'EncoderDecoderCache' object is
                                  # not subscriptable" inside
                                  # modeling_florence2.py.
            )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
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
