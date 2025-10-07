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

"""
PYTHONPATH=. streamlit run cosmos_transfer2/_src/predict2/inference/text2image.py --server.port 2222
"""

import argparse

import streamlit as st
import torch

from cosmos_transfer2._src.predict2.datasets.utils import IMAGE_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

torch.enable_grad(False)

DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_POSITIVE_PROMPT = (
    "filmic photo of a group of three women on a street downtown, they are holding their hands up the camera"
)

# 2B model
DEFAULT_S3_CHECKPOINT_DIR = "s3://bucket/cosmos_diffusion_v2/official_runs_reason_embeddings/official_runs_reason_embeddings_028_2B_1024res_pretrain_synthetic_photoreal_prompted_mix_qwen_7b_vl_crossattn_proj_full_concat_hq_tuning/checkpoints/iter_000030000/"
DEFAULT_EXPERIMENT_NAME = "official_runs_reason_embeddings_028_2B_1024res_pretrain_synthetic_photoreal_prompted_mix_qwen_7b_vl_crossattn_proj_full_concat_hq_tuning"  # Example experiment name, adjust if needed

# 14B model
# DEFAULT_S3_CHECKPOINT_DIR = "s3://bucket/cosmos_diffusion_v2/official_runs_reason_embeddings/official_runs_reason_embeddings_111_14B_1024res_pretrain_synthetic_photoreal_mix_projection_full_concat_qwen_7b_vl_hq_tuning/checkpoints/iter_000030000/"
# DEFAULT_EXPERIMENT_NAME = (
#     "official_runs_reason_embeddings_111_14B_1024res_pretrain_synthetic_photoreal_mix_projection_full_concat_qwen_7b_vl_hq_tuning"
# )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="simple text2world inference script")
    parser.add_argument("--experiment", type=str, default="???", help="inference only config")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance value")
    parser.add_argument(
        "--s3_checkpoint_dir",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--prompt", type=str, default=DEFAULT_POSITIVE_PROMPT, help="Prompt for the video")
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Prompt for the video")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    return parser.parse_args()


def get_sample_batch(
    resolution: str = "1024",
    aspect_ratio: str = "16,9",
    batch_size: int = 1,
) -> torch.Tensor:
    w, h = IMAGE_RES_SIZE_INFO[resolution][aspect_ratio]
    data_batch = {
        "dataset_name": "image_data",
        "images": torch.randn(batch_size, 3, h, w).cuda(),
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda(),
        "fps": torch.randint(16, 32, (batch_size,)).cuda(),
        "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
    }

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch


class Text2ImageInference:
    def __init__(self, experiment_name: str, ckpt_path: str, s3_credential_path: str):
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path

        model, config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            config_file="cosmos_transfer2/_src/predict2/configs/text2world/config.py",
            s3_checkpoint_dir=ckpt_path,
            enable_fsdp=False,
            load_ema_to_reg=True,
        )
        self.model = model
        self.config = config
        self.resolution = str(self.model.config.resolution)  # Store resolution from loaded model

    def generate_image(
        self, prompt: str, neg_prompt: str, guidance: int = 7, aspect_ratio: str = "16,9", num_samples: int = 1
    ):
        data_batch = get_sample_batch(
            resolution=self.resolution,  # Use resolution from loaded model
            aspect_ratio=aspect_ratio,
            batch_size=num_samples,
        )

        # modify the batch if prompt is provided
        if self.model.text_encoder is not None:
            # Text encoder is defined in the model class. Use it
            if prompt:
                data_batch["ai_caption"] = [prompt]
                data_batch["t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [prompt], "images": None},
                    input_caption_key="ai_caption",
                )
            if neg_prompt:
                data_batch["neg_t5_text_embeddings"] = self.model.text_encoder.compute_text_embeddings_online(
                    data_batch={"ai_caption": [neg_prompt], "images": None},
                    input_caption_key="ai_caption",
                )
        else:
            if prompt:
                text_emb = get_text_embedding(prompt)
                data_batch["t5_text_embeddings"] = text_emb.to(dtype=torch.bfloat16).cuda()
            if neg_prompt:
                text_emb = get_text_embedding(neg_prompt)
                data_batch["neg_t5_text_embeddings"] = text_emb.to(dtype=torch.bfloat16).cuda()

        # generate samples
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            seed=torch.randint(0, 10000, (1,)).item(),  # Use random seed for variation
            is_negative_prompt=bool(neg_prompt),  # Only set true if neg_prompt provided
        )
        out_samples = self.model.decode(sample)
        out_samples = (1.0 + out_samples) / 2  # Convert from [-1, 1] to [0, 1]
        out_samples = out_samples.clamp(0, 1)  # Clamp values
        out_samples = out_samples.squeeze(2)  # Convert the video tensor to image tensor

        # Now reshape
        return out_samples


# Cache the model loading based on the checkpoint path
@st.cache_resource
def get_inference_model(experiment_name, s3_checkpoint_dir, s3_cred):
    print(f"Loading model from {s3_checkpoint_dir}...")  # Add print statement for debugging
    # Setup S3 backend here if needed by easy_io implicitly used in loader
    try:
        from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io

        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "s3_credential_path": s3_cred,
            }
        )
        print("S3 backend set.")
    except ImportError:
        st.warning(
            "easy_io not found, S3 backend setup skipped. Model loading might fail if using S3 paths without explicit credentials."
        )
    except Exception as e:
        st.error(f"Error setting S3 backend: {e}")
        # Decide if we should proceed or stop

    try:
        model_instance = Text2ImageInference(experiment_name, s3_checkpoint_dir, s3_cred)
        print("Model loaded successfully.")
        return model_instance
    except Exception as e:
        st.error(f"Error loading model from {s3_checkpoint_dir}: {e}")
        return None  # Return None or raise to stop the app


def streamlit_main():
    st.set_page_config(layout="wide")
    st.title("🎨 Text-to-Image Generation Demo")

    # --- Sidebar Inputs ---
    st.sidebar.header("Model Configuration")
    # Use the actual experiment name if it varies per checkpoint
    exp_name = st.sidebar.text_input("Experiment Name (usually part of path):", value=DEFAULT_EXPERIMENT_NAME)
    s3_dir = st.sidebar.text_input("S3 Checkpoint Directory:", value=DEFAULT_S3_CHECKPOINT_DIR)
    s3_cred_path = st.sidebar.text_input("S3 Credentials Path:", value="credentials/s3_checkpoint.secret")

    st.sidebar.header("Generation Parameters")
    prompt = st.sidebar.text_area("Prompt:", value=DEFAULT_POSITIVE_PROMPT, height=150)
    neg_prompt = st.sidebar.text_area("Negative Prompt:", value=DEFAULT_NEGATIVE_PROMPT, height=150)
    guidance = st.sidebar.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=7.0, step=0.5)

    # --- Load Model --- (Attempt only if path is provided)
    model_instance = None
    aspect_ratio_options = ["1,1"]  # Start with default, update after model load

    if s3_dir and exp_name:
        with st.spinner("Loading Text2Image model..."):
            model_instance = get_inference_model(exp_name, s3_dir, s3_cred_path)

        if model_instance:
            st.sidebar.success("Model loaded successfully!")
            # Get available aspect ratios for the loaded model's resolution
            try:
                res_str = model_instance.resolution
                aspect_ratio_options = list(IMAGE_RES_SIZE_INFO.get(res_str, {"1,1": None}).keys())
            except Exception as e:
                st.sidebar.error(f"Could not get aspect ratios for resolution {model_instance.resolution}: {e}")
                aspect_ratio_options = ["1,1"]  # Fallback
        else:
            st.sidebar.error("Failed to load model. Check path and credentials.")

    aspect_ratio = st.sidebar.selectbox("Aspect Ratio:", options=aspect_ratio_options, index=0)
    num_samples = st.sidebar.number_input("Number of Samples:", min_value=1, max_value=8, value=1, step=1)
    # --- Main Area ---
    generate_button = st.button("Generate Image(s)", disabled=(model_instance is None))

    if generate_button and model_instance:
        st.subheader("Generated Images")
        with st.spinner("Generating images..."):
            try:
                out_samples_tensor = model_instance.generate_image(
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    guidance=guidance,
                    aspect_ratio=aspect_ratio,
                    num_samples=num_samples,
                )

                # Display the generated images
                num_cols = min(num_samples, 4)  # Display up to 4 images per row
                cols = st.columns(num_cols)
                for i in range(num_samples):
                    with cols[i % num_cols]:
                        out_sample_i = out_samples_tensor[i].cpu().permute(1, 2, 0).numpy()
                        st.image(out_sample_i, caption=f"Sample {i + 1}", use_column_width=False)

            except Exception as e:
                st.error(f"Error during image generation: {e}")
    elif generate_button and model_instance is None:
        st.error("Model not loaded. Cannot generate images.")


if __name__ == "__main__":
    streamlit_main()
