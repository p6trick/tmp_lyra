"""Qwen3-VL backend for image-plus-text greedy generation."""

import copy
import json

from .config import InferenceConfig, gpu_dtype


class QwenBackend:
    def __init__(self, source: dict, settings: InferenceConfig):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

        dtype = gpu_dtype(torch, settings.dtype)
        set_seed(settings.seed)
        revision = {"revision": source["commit"]} if source["commit"] else {}
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(source["model"], **revision)
        self.processor.image_processor.size = {
            "shortest_edge": settings.min_pixels, "longest_edge": settings.max_pixels,
        }
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            source["model"], **revision, dtype=dtype,
            attn_implementation=settings.attention, device_map={"": 0},
        ).eval()
        self.generation = copy.deepcopy(self.model.generation_config)
        self.generation.max_new_tokens = settings.max_new_tokens
        self.generation.do_sample = False
        self.generation.num_beams = 1
        self.generation.num_return_sequences = 1
        self.generation.return_dict_in_generate = False
        self.generation.temperature = 1.0
        self.generation.top_p = 1.0
        self.generation.top_k = 50
        self.generation_settings = json.loads(self.generation.to_json_string(use_diff=False))
        self.runtime = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        }

    def render(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        return self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

    def generate(self, image, prompts: list[str]) -> list[str]:
        inputs = self.processor(
            images=[image] * len(prompts), text=prompts, padding=True,
            return_tensors="pt", add_special_tokens=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=self.generation)
        continuation = result[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

    def clear_cache(self) -> None:
        self.torch.cuda.empty_cache()
