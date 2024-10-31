import os
from typing import Any, Union

import pandas as pd
from tqdm import tqdm
from transformers import set_seed
import torch
import openvino_genai

from .registry import register_evaluator, BaseEvaluator

from .whowhat_metrics import ImageSimilarity

default_data = {
    "prompts": [
        "Cinematic, a vibrant Mid-century modern dining area, colorful chairs and a sideboard, ultra realistic, many detail",
        "colibri flying near a flower, side view, forest background, natural light, photorealistic, 4k",
        "Illustration of an astronaut sitting in outer space, moon behind him",
        "A vintage illustration of a retro computer, vaporwave aesthetic, light pink and light blue",
        "A view from beautiful alien planet, very beautiful, surealism, retro astronaut on the first plane, 8k photo",
        "red car in snowy forest, epic vista, beautiful landscape, 4k, 8k",
        "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
        "cute cat 4k, high-res, masterpiece, best quality, soft lighting, dynamic angle",
        "A cat holding a sign that says hello OpenVINO",
        "A small cactus with a happy face in the Sahara desert.",
    ],
}


class Generator(openvino_genai.Generator):
    def __init__(self, seed, rng, mu=0.0, sigma=1.0):
        openvino_genai.Generator.__init__(self)
        self.mu = mu
        self.sigma = sigma
        self.rng = rng

    def next(self):
        return torch.randn(1, generator=self.rng, dtype=torch.float32).item()


@register_evaluator("text-to-image")
class Text2ImageEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "openai/clip-vit-large-patch14",
        resolution=(512, 512),
        num_inference_steps=4,
        crop_prompts=True,
        num_samples=None,
        gen_image_fn=None,
        seed=42,
        is_genai=False,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.resolution = resolution
        self.crop_prompt = crop_prompts
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.similarity = None
        self.similarity = ImageSimilarity(similarity_model_id)
        self.last_cmp = None
        self.gt_dir = os.path.dirname(gt_data)
        self.generation_fn = gen_image_fn
        self.is_genai = is_genai

        if base_model:
            base_model.resolution = self.resolution
            self.gt_data = self._generate_data(
                base_model, gen_image_fn, os.path.join(self.gt_dir, "reference")
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

    def get_generation_fn(self):
        return self.generation_fn

    def dump_gt(self, csv_name: str):
        self.gt_data.to_csv(csv_name)

    def score(self, model, gen_image_fn=None):
        model.resolution = self.resolution
        predictions = self._generate_data(
            model, gen_image_fn, os.path.join(self.gt_dir, "target")
        )

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["images"].values
        self.last_cmp["optimized_model"] = predictions["images"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        res = self.last_cmp.nsmallest(top_k, metric)
        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_image_fn=None, image_dir="reference"):
        if hasattr(model, "reshape") and self.resolution is not None:
            if gen_image_fn is None:
                model.reshape(
                    batch_size=1,
                    height=self.resolution[0],
                    width=self.resolution[1],
                    num_images_per_prompt=1,
                )

        def default_gen_image_fn(model, prompt, num_inference_steps, generator=None):
            output = model(
                prompt,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                width=self.resolution[0],
                height=self.resolution[0],
                generator=generator,
            )
            return output.images[0]

        generation_fn = gen_image_fn or default_gen_image_fn

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"prompts": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame.from_dict(default_data)

        prompts = data["prompts"]
        prompts = (
            prompts.values
            if self.num_samples is None
            else prompts.values[: self.num_samples]
        )
        images = []
        rng = torch.Generator(device="cpu")

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for i, prompt in tqdm(enumerate(prompts), desc="Evaluate pipeline"):
            set_seed(self.seed)
            rng = rng.manual_seed(self.seed)
            image = generation_fn(
                model,
                prompt,
                self.num_inference_steps,
                generator=Generator(self.seed, rng) if self.is_genai else rng
            )
            image_path = os.path.join(image_dir, f"{i}.png")
            image.save(image_path)
            images.append(image_path)

        res_data = {"prompts": list(prompts), "images": images}
        df = pd.DataFrame(res_data)

        return df
