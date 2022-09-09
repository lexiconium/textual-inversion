import os
import random
from collections import namedtuple
from functools import partial
from typing import Iterable, List

import torchvision
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

SUPPORTED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


def has_supported_image_extension(
    filename: str, supported_extensions: Iterable[str]
):
    return filename.split(".")[-1] in supported_extensions


IMAGENET_TEMPLATES_SMALL = {
    "object": [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}"
    ],
    "style": [
        "a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}"
    ]
}

TextualInversionData = namedtuple(
    "TextualInversionData",
    field_names=["pixel_values", "input_ids"]
)


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: torchvision.transforms.Compose,
        tokenizer: CLIPTokenizer,
        placeholder_token: str,
        learnable_property: str
    ):
        self.image_paths = [
            os.path.join(data_dir, filename)
            for filename in filter(
                partial(
                    has_supported_image_extension,
                    supported_extensions=SUPPORTED_IMAGE_EXTENSIONS
                ),
                os.listdir(data_dir)
            )
        ]

        self.tokenize = partial(
            tokenizer,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )

        self.templates = IMAGENET_TEMPLATES_SMALL[learnable_property]

        self.data_dir = data_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.learn_type = learnable_property

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        pixel_values = self.transforms(image)

        template = random.choice(self.templates)
        text = template.format(self.placeholder_token)

        input_ids = self.tokenize(text).input_ids[0]

        return TextualInversionData(
            pixel_values=pixel_values, input_ids=input_ids
        )


class RepeatedDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, num_repeat: int):
        self.dataset = dataset
        self.num_repeat = num_repeat

    def __len__(self):
        return len(self.dataset) * self.num_repeat

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]

    def unwrap(self):
        return self.dataset
