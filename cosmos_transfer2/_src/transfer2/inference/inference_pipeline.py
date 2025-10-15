# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
from typing import Optional, Union

import torch

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_transfer2._src.transfer2.inference.utils import (
    get_t5_from_prompt,
    normalized_float_to_uint8,
    read_and_process_control_input,
    read_and_process_image_context,
    read_and_process_video,
    reshape_output_video_to_input_resolution,
    uint8_to_normalized_float,
)


class ControlVideo2WorldInference:
    """
    Handles the Control2Video inference process, including model loading, data preparation,
    and video transfer from an input video and text prompt.
    """

    def __init__(
        self,
        registered_exp_name: str,
        checkpoint_paths: Union[str, list[str]],
        s3_credential_path: str,
        exp_override_opts: Optional[list[str]] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        cache_dir: Optional[str] = None,
        skip_load_model: bool = False,
        base_load_from: Optional[str] = None,
    ):
        """
        Initializes the ControlVideo2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path.

        Args:
            registered_exp_name (str): Name of the experiment configuration.
            checkpoint_paths (Union[str, list[str]]): Single checkpoint path or List of checkpoint paths for multi-branch models.
            s3_credential_path (str): Path to S3 credentials file for ckpt & negative embedding (if loading from S3).
            exp_override_opts (list[str]): List of experiment override options.
            process_group (torch.distributed.ProcessGroup): Process group for distributed training.
            cache_dir (str): Cache directory for storing pre-computed embeddings.
            skip_load_model (bool): Whether to skip loading model from checkpoint for multi-control models.
        """
        self.registered_exp_name = registered_exp_name
        self.checkpoint_path = checkpoint_paths if isinstance(checkpoint_paths, str) else checkpoint_paths[0]
        self.s3_credential_path = s3_credential_path
        self.cache_dir = cache_dir

        if exp_override_opts is None:
            exp_override_opts = []
        # no need to load base model separately at inference
        exp_override_opts.append("model.config.base_load_from=null")

        # Load the model and config. Each trained model's config is composed by
        # loading a pre-registered experiment config, and then (optionally) overriding with some command-line
        # arguments. That is done in experiment_list.py. Here we simply replicate that process.
        model, config = load_model_from_checkpoint(
            experiment_name=self.registered_exp_name,
            s3_checkpoint_dir=self.checkpoint_path,
            config_file="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py",
            load_ema_to_reg=True,
            local_cache_dir=(
                cache_dir if not checkpoint_paths else None
            ),  # for multi-control models, need to load other branches before caching
            experiment_opts=exp_override_opts,
        )
        if (
            isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1 and not skip_load_model
        ):  # load other branches for multi-control models
            load_from_local = False
            if cache_dir is not None:
                # build a unique path for s3checkpoint dir
                local_s3_ckpt_fp = os.path.join(
                    cache_dir,
                    self.checkpoint_path.split("s3://")[1],
                    "torch_model",
                    f"_rank_{distributed.get_rank()}.pt",
                )
                if os.path.exists(local_s3_ckpt_fp):
                    load_from_local = True

            if load_from_local:
                log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
                model.load_state_dict(easy_io.load(local_s3_ckpt_fp))
            else:
                model.load_multi_branch_checkpoints(checkpoint_paths=checkpoint_paths)
                if cache_dir is not None:
                    log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
                    easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

        if base_load_from is not None:
            log.info(f"Loading base model from {base_load_from}")
            model.config.base_load_from = {
                "load_path": base_load_from,
                "credentials": s3_credential_path,
            }
            model.load_base_model()

        self.text_encoder_class = model.text_encoder_class

        if process_group is not None:
            log.info("Enabling CP in base model\n")
            model.net.enable_context_parallel(process_group)

        self.model = model
        self.config = config
        self.batch_size = 1

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prev_output: torch.Tensor,
        text_embedding: torch.Tensor,
        fps: int,
        negative_prompt: str = None,
        control_weight: str = "1.0",
        image_context: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.

            image_context (torch.Tensor, optional): Image context tensor for conditioning. Can be (B, C, H, W).

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = prev_output.shape
        input_key = "video" if T > 1 else "images"

        data_batch = {
            "dataset_name": "video_data",
            input_key: prev_output.squeeze(2),
            "t5_text_embeddings": text_embedding,  # positive prompt embedding. Name has t5 but also supports Reason1.
            "fps": torch.randint(16, 32, (self.batch_size,)).cuda(),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W).cuda(),  # Padding mask (assumed no padding here)
            "num_conditional_frames": 1,  # Specify that the first frame is conditional
            "control_weight": [float(w) for w in control_weight.split(",")],
            "input_video": video,
        }

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        # Add image context
        if image_context is not None:
            data_batch["image_context"] = image_context.cuda().to(dtype=torch.bfloat16).contiguous()

        # Handle negative prompts for classifier-free guidance
        if negative_prompt is not None:
            assert self.neg_t5_embeddings is not None, "Negative prompt embedding is not computed."
            data_batch["neg_t5_text_embeddings"] = self.neg_t5_embeddings

        return data_batch

    def _get_num_chunks(
        self, input_frames: torch.Tensor, num_video_frames_per_chunk: int, num_conditional_frames: int
    ) -> tuple[int, int, int]:
        """
        Get the number of chunks for chunk-wise long video generation.
        """
        # Frame number settting for chunk-wise long video generation
        num_total_frames = input_frames.shape[1]
        num_frames_per_chunk = num_video_frames_per_chunk - num_conditional_frames
        if num_video_frames_per_chunk == 1:
            num_chunks = 1
        else:
            num_generated_frames_vid2vid = num_total_frames - num_video_frames_per_chunk
            num_chunks = 1 + num_generated_frames_vid2vid // num_frames_per_chunk
            if num_generated_frames_vid2vid % num_frames_per_chunk != 0:
                num_chunks += 1

        return num_total_frames, num_chunks, num_frames_per_chunk

    def _pad_input_frames(
        self,
        input_frames: torch.Tensor,
        num_total_frames: int,
        num_video_frames_per_chunk: int,
        padding_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Pad input frames if total frames is less than chunk size
        """
        if num_total_frames < num_video_frames_per_chunk:
            if padding_mode == "repeat":
                last_frame = input_frames[:, -1:, :, :]  # Get the last frame
                padding = last_frame.repeat(1, num_video_frames_per_chunk - num_total_frames, 1, 1)
                input_frames = torch.cat([input_frames, padding], dim=1)
            elif padding_mode == "reflect":
                while input_frames.shape[1] < num_video_frames_per_chunk:
                    padding = min(input_frames.shape[1] - 1, num_video_frames_per_chunk - input_frames.shape[1])
                    padding_frames = input_frames.flip(dims=[1])[:, :padding, :, :]
                    input_frames = torch.cat([input_frames, padding_frames], dim=1)
            else:
                raise ValueError(f"Invalid padding mode: {padding_mode}")
        return input_frames

    @torch.no_grad()
    def generate_img2world(
        self,
        prompt: str | torch.Tensor | list[str] | dict[str, str],
        video_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93,
        num_steps: int = 35,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        input_control_video_paths: dict[str, str] | None = None,
        show_control_condition: bool = False,
        show_input: bool = False,
        image_context_path: Optional[str] = None,
        keep_input_resolution: bool = True,
        negative_prompt: str | None = None,
        max_frames: int | None = None,
        context_frame_idx: int | None = None,
    ) -> tuple[torch.Tensor, int, tuple[int, int]]:
        """
        Generates a video based on an input video and text prompt.
        Supports chunk-wise long video generation.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            video_path (str): Path to the input conditional video.
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            resolution (str, optional): Resolution of the video (720-default, 480, etc). Defaults to 720.
            image_context_path (str, optional): Path to image file to use as image context. If None, uses random frame from video. Will be ignored and use input video if context_frame_idx is provided.
            keep_input_resolution (bool, optional): Whether to keep the exact dimension of the. Defaults to True.
            negative_prompt (str, optional): Negative prompt for classifier-free guidance. Defaults to None.
            max_frames (int, optional): Maximum number of frames to read from the video. Defaults to None. 1 for image.
            context_frame_idx (int, optional): Frame index of the input video to use as image context. Defaults to None. In this case, can still use image_context_path to provide image context.
        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
            int: Frames per second of the original input video.
            tuple[int, int]: Original height and width of the input video.

        Raises:
            ValueError: If the input video is empty or invalid.
        """
        # --------Input processing--------
        # Process input video and get meta info.
        log.info("Loading input video...")
        # aspect_ratio is width / height
        # input_frames is (C, T, H, W)
        input_frames, fps, aspect_ratio, original_hw = read_and_process_video(
            video_path, resolution=resolution, max_frames=max_frames
        )
        if input_frames.shape[1] == 0:
            raise ValueError("Input video is empty")

        # Get text context embeddings
        log.info("Computing prompt text embeddings...")
        if self.text_encoder_class == "T5":
            text_embeddings = get_t5_from_prompt(prompt, text_encoder_class="T5", cache_dir=self.cache_dir)
        else:
            text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
            )
        if negative_prompt:
            log.info("Computing negative prompt text embeddings...")
            if self.text_encoder_class == "T5":
                neg_text_embeddings = get_t5_from_prompt(
                    negative_prompt, text_encoder_class="T5", cache_dir=self.cache_dir
                )
            else:
                neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    {"ai_caption": [negative_prompt], "images": None}, input_caption_key="ai_caption"
                )
            self.neg_t5_embeddings = neg_text_embeddings

        # Process image context if provided; else will be None
        log.info("Processing image context if available...")
        if context_frame_idx is not None:
            image_context_path = video_path
            log.info(f"Using context frame index: {context_frame_idx} from video path: {video_path}")
        image_context = read_and_process_image_context(
            image_context_path,
            resolution=(VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]),
            resize=True,
            context_frame_idx=context_frame_idx,
        )
        # Load control inputs from paths, or optionally compute on-the-fly, and add to data batch.
        log.info("Loading control inputs...")
        control_input_dict = read_and_process_control_input(
            input_control_video_paths, hint_key=hint_key, resolution=resolution
        )

        # -------- Stuff to handle chunk-wise long video generation --------
        num_total_frames, num_chunks, num_frames_per_chunk = self._get_num_chunks(
            input_frames, num_video_frames_per_chunk, num_conditional_frames
        )
        # Pad input frames if total frames is less than chunk size
        input_frames = self._pad_input_frames(input_frames, num_total_frames, num_video_frames_per_chunk)
        all_chunks, time_per_chunk = [], []
        # For first chunk, use zeros as input (after normalization it is 0)
        prev_output = torch.zeros_like(input_frames[:, :num_video_frames_per_chunk]).to(torch.uint8).cuda()[None]

        # --------Start of chunk-wise long video generation--------
        for chunk_id in range(num_chunks):
            log.info(f"Generating chunk {chunk_id + 1}/{num_chunks}")
            start_time = time.perf_counter()

            # Calculate start frame for this chunk
            chunk_start_frame = chunk_id * num_frames_per_chunk
            chunk_end_frame = min(chunk_start_frame + num_video_frames_per_chunk, input_frames.shape[1])

            x_sigma_max = None
            if input_frames is not None:
                cur_input_frames = input_frames[:, chunk_start_frame:chunk_end_frame]
                cur_input_frames = self._pad_input_frames(
                    cur_input_frames, cur_input_frames.shape[1], num_video_frames_per_chunk
                )
                if sigma_max is not None:
                    x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None].cuda()
                    x0 = self.model.encode(x0).contiguous()
                    x_sigma_max = self.model.get_x_from_clean(x0, sigma_max, seed=(seed + chunk_id))

            if isinstance(text_embeddings, list):
                text_emb_idx = min(chunk_id, len(text_embeddings) - 1)
                text_embedding = text_embeddings[text_emb_idx]
            else:
                text_embedding = text_embeddings

            # Prepare the data batch with current input. Note: this doesn't include control inputs yet.
            data_batch = self._get_data_batch_input(
                cur_input_frames,
                prev_output,
                text_embedding,
                fps,
                negative_prompt=negative_prompt,
                control_weight=control_weight,
                image_context=image_context,
            )

            # Process control inputs as specified in the hint_key list.
            # If pre-computed control inputs are provided, load them into the data batch.
            # Online computation of depth/seg also happens here (if needed).
            for k, v in control_input_dict.items():
                cur_control_input = v[:, chunk_start_frame:chunk_end_frame]
                data_batch[k] = self._pad_input_frames(
                    cur_control_input, cur_control_input.shape[1], num_video_frames_per_chunk
                )
                if k == "control_input_inpaint_mask":
                    data_batch["control_input_inpaint"] = cur_input_frames
            # Otherwise, compute control inputs on-the-fly via the augmentor（applicable to edge and vis).
            data_batch = get_augmentor_for_eval(
                data_dict=data_batch,
                input_keys=["input_video"],
                output_keys=hint_key,
                preset_edge_threshold=preset_edge_threshold,
                preset_blur_strength=preset_blur_strength,
            )

            if chunk_id == 0:
                data_batch[NUM_CONDITIONAL_FRAMES_KEY] = 0
            else:
                data_batch[NUM_CONDITIONAL_FRAMES_KEY] = (
                    1 + (num_conditional_frames - 1) // 4
                )  # tokenizer temporal compression is 4x

            random.seed(seed)
            seed = random.randint(0, 1000000)
            log.info(f"Seed: {seed}")

            # Generate and decode video
            sample = self.model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                guidance=guidance,
                seed=seed,
                is_negative_prompt=negative_prompt is not None,
                x_sigma_max=x_sigma_max,
                sigma_max=sigma_max,
                num_steps=num_steps,
            )
            video = self.model.decode(sample).cpu()  # Shape: (1, C, T, H, W)

            # For visualization: concatenate condition and input videos with generated video
            if show_control_condition:
                # Tensor value range: [-1, 1]
                # From top to bottom: input_video, control_input, generated_video
                conditions = []
                # if input_video is not None:
                #     conditions += [input_video[:, :, start_frame:end_frame]]
                for key in hint_key:
                    control_input = data_batch["control_input_" + key]
                    if f"control_input_{key}_mask" in data_batch:
                        control_input = (control_input + 1) / 2 * data_batch[f"control_input_{key}_mask"] * 2 - 1
                    conditions += [control_input.cpu()]
                video_cat = torch.cat([*conditions, video], dim=-1)
            elif show_input and input_frames is not None:
                x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None]
                video_cat = torch.cat([x0, video], dim=-1)
            else:
                video_cat = video

            if chunk_id == 0:
                all_chunks.append(video_cat.cpu())
            else:
                # For subsequent chunks, only append the non-overlapping frames
                all_chunks.append(video_cat[:, :, num_conditional_frames:, :, :].cpu())

            # For next chunk, use last conditional_frames as input
            if chunk_id < num_chunks - 1:  # Don't need to prepare next input for last chunk
                last_frames = video[:, :, -num_conditional_frames:, :, :]  # (1, C, num_conditional_frames, H, W)
                # Convert to uint8 [0, 255]
                last_frames_uint8 = normalized_float_to_uint8(last_frames)
                # Create blank frames for the rest
                blank_frames = torch.zeros(
                    (
                        1,
                        3,
                        num_video_frames_per_chunk - num_conditional_frames,
                        video.shape[-2],
                        video.shape[-1],
                    ),
                    dtype=torch.uint8,
                    device=video.device,
                )
                prev_output = torch.cat([last_frames_uint8, blank_frames], dim=2)
            end_time = time.perf_counter()
            time_per_chunk.append(end_time - start_time)

        # Concatenate all chunks along time
        full_video = torch.cat(all_chunks, dim=2)  # (1, C, T, H, W)
        # Keep only the original number of frames
        full_video = full_video[:, :, :num_total_frames, :, :]

        if keep_input_resolution:
            # reshape output video to match the input video resolution
            full_video = reshape_output_video_to_input_resolution(
                full_video, hint_key, show_control_condition, show_input, original_hw
            )
        log.info(f"Average time per chunk: {sum(time_per_chunk) / len(time_per_chunk)}")
        return full_video, fps, original_hw
