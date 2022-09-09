import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

import PIL
import numpy as np
import torch.optim
from PIL import Image
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    get_scheduler
)
from diffusers.utils import BaseOutput
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .dataset import RepeatedDatasetWrapper, TextualInversionDataset


@dataclass
class TextualInversionDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class TextualInversionDiffusionTrainingConfig:
    placeholder_token: str = field(metadata={"help": ""})
    initializer_token: str = field(metadata={"help": ""})
    num_training_epochs: int = field(metadata={"help": ""})
    training_batch_size: int = field(metadata={"help": ""})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": ""})
    learning_rate: float = field(default=1e-4, metadata={"help": ""})
    adam_beta1: float = field(default=0.9, metadata={"help": ""})
    adam_beta2: float = field(default=0.999, metadata={"help": ""})
    adam_weight_decay: float = field(default=1e-2, metadata={"help": ""})
    adam_epsilon: float = field(default=1e-8, metadata={"help": ""})
    lr_scheduler: str = field(default="constant", metadata={"help": ""})
    warmup_ratio: float = field(default=0, metadata={"help": ""})
    num_warmup_steps: int = field(default=0, metadata={"help": ""})
    mixed_precision: str = field(default="no", metadata={"help": ""})
    seed: int = field(default=42, metadata={"help": ""})
    output_dir: str = field(default="outputs", metadata={"help": ""})

    def __post_init__(self):
        template = "Invalid {} = {} found. Will be set to {}."

        if self.num_training_epochs < 1:
            warnings.warn(template.format("num_training_epochs", self.num_training_epochs, 1))
            self.num_training_epochs = 1

        if self.training_batch_size < 1:
            warnings.warn(template.format("training_batch_size", self.training_batch_size, 1))
            self.training_batch_size = 1

        if self.gradient_accumulation_steps < 1:
            warnings.warn(template.format("gradient_accumulation_steps", self.gradient_accumulation_steps, 1))
            self.gradient_accumulation_steps = 1

        if self.learning_rate < 0:
            warnings.warn(template.format("learning_rate", self.learning_rate, 1e-4))
            self.learning_rate = 1e-4


class TextualInversionDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~textual_inversion.TextualInversionDiffusionPipelineOutput`] or `tuple`:
            [`~textual_inversion.TextualInversionDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return image

        return TextualInversionDiffusionPipelineOutput(images=image)

    def _freeze(self):
        """
        Freezes model parameters except token embeddings.
        """

        def freeze_parameters(model: nn.Module, exclude: Union[None, str, Iterable] = None):
            if exclude is None:
                param_iterator = model.named_parameters()
            elif isinstance(exclude, str):
                param_iterator = filter(
                    lambda named_parameter: named_parameter[0].split(".")[-2] != exclude,
                    model.named_parameters()
                )
            elif isinstance(exclude, Iterable):
                param_iterator = filter(
                    lambda named_parameter: named_parameter[0].split(".")[-2] not in exclude,
                    model.named_parameters()
                )
            else:
                raise ValueError(f"Argument `exclude` must be a str or an iterable. Found {type(exclude)} instead.")

            for _, param in param_iterator:
                param.requires_grad = False

        freeze_parameters(self.text_encoder, exclude="token_embedding")
        freeze_parameters(self.unet)
        freeze_parameters(self.vae)

    def _mode(self, train: bool):
        if train:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

        self.unet.eval()
        self.vae.eval()

    def train(
        self,
        training_config: TextualInversionDiffusionTrainingConfig,
        dataset: TextualInversionDataset
    ):
        # Add placeholder to the tokenizer after checking initializer and placeholder tokens
        initializer_token_ids = self.tokenizer.encode(training_config.initializer_token, add_special_tokens=False)
        if len(initializer_token_ids) > 1:
            raise ValueError("Initializer token must be a single token.")

        num_added_tokens = self.tokenizer.add_tokens(training_config.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"Tokenizer already contains the token {training_config.placeholder_token}."
                " Please pass a different placeholder token that is not already in the tokenizer."
            )

        initializer_token_id = initializer_token_ids[0]
        placeholder_token_id = self.tokenizer.convert_tokens_to_ids(training_config.placeholder_token)

        # Resize token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize placeholder token embedding with that of initializer token
        token_embeddings = self.text_encoder.get_input_embeddings()
        _token_embeddings = token_embeddings.weight.data
        _token_embeddings[placeholder_token_id] = _token_embeddings[initializer_token_id]

        # Freeze parameters except token embeddings
        self._freeze()

        # Set optimizer and learning rate scheduler
        optimizer = optim.AdamW(
            token_embeddings.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            weight_decay=training_config.adam_weight_decay,
            eps=training_config.adam_epsilon
        )

        drop_last = False
        rounding_fn = int if drop_last else math.ceil

        num_training_images = training_config.num_training_epochs * len(dataset)
        images_per_step = training_config.training_batch_size * training_config.gradient_accumulation_steps
        num_training_steps = rounding_fn(num_training_images / images_per_step)

        num_warmup_steps = training_config.num_warmup_steps
        if not num_warmup_steps and training_config.warmup_ratio:
            num_warmup_steps = int(num_training_steps * training_config.warmup_ratio)

        lr_scheduler = get_scheduler(
            training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Wrap dataset for easy implementation of Accelerator
        dataset = RepeatedDatasetWrapper(dataset, num_repeat=training_config.num_training_epochs)

        # Set dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.training_batch_size,
            shuffle=True,
            drop_last=drop_last
        )

        # Set accelerator
        accelerator = Accelerator(
            mixed_precision=training_config.mixed_precision,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            step_scheduler_with_optimizer=True
        )
        self.text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(
            self.text_encoder, optimizer, lr_scheduler, dataloader
        )
        self.to(accelerator.device)

        # Set to training mode
        self._mode(train=True)

        progress_bar = tqdm(
            range(num_training_steps),
            desc="Training step",
            disable=not accelerator.is_local_main_process
        )

        for batch in dataloader:
            with accelerator.accumulate(self.text_encoder):
                optimizer.zero_grad()

                latent_dists = self.vae.encode(batch.pixel_values).latent_dist
                latents: torch.Tensor = 0.18215 * latent_dists.sample().detach()

                # Sample noise
                noises = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device)
                timesteps = torch.randint(
                    len(self.scheduler),
                    size=(latents.shape[0],),
                    dtype=torch.long,
                    device=latents.device
                )

                # Add noise to the latent vectors
                noisy_latents = self.scheduler.add_noise(latents, noises, timesteps)

                # Get text embedding for conditioning
                encoder_hidden_states = self.text_encoder(batch.input_ids).last_hidden_state

                # Predict noise residual
                noise_preds = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(noise_preds, noises, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out gradients other than placeholder token embedding
                if accelerator.num_processes == 1:
                    grads = self.text_encoder.get_input_embeddings().weight.grad
                else:
                    grads = self.text_encoder.module.get_input_embeddings().weight.grad

                grads.data[torch.arange(len(self.tokenizer)) != placeholder_token_id].fill_(0)

                optimizer.step()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update()

        # Set to evaluation mode
        self._mode(train=False)

        return self
