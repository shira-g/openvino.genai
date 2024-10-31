#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino_genai
from PIL import Image
import numpy as np

class Generator(openvino_genai.Generator):
    def __init__(self, seed, mu=0.0, sigma=1.0):
        openvino_genai.Generator.__init__(self)
        np.random.seed(seed)
        self.mu = mu
        self.sigma = sigma

    def next(self):
        return np.random.normal(self.mu, self.sigma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)

    image_tensor = pipe.generate(
        args.prompt,
        width=512,
        height=512,
        num_inference_steps=20,
        num_images_per_prompt=1,
        generator=Generator(42)  # openvino_genai.CppStdGenerator can be used to have same images as C++ sample
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")


if '__main__' == __name__:
    main()
