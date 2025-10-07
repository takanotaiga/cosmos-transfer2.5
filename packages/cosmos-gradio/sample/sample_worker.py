from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_worker import ModelWorker
import os
from PIL import Image


class SampleWorker(ModelWorker):
    def __init__(self, num_gpus, checkpoint_dir, model_name):
        pass

    def infer(self, args: dict):
        output_dir = args.get("output_dir", "/mnt/pvc/gradio_output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prompt = args.get("prompt", "")

        img = Image.new("RGB", (256, 256), color="red")
        out_file_name = os.path.join(output_dir, "output.png")
        img.save(out_file_name)

        return {"message": "created a red box", "prompt": prompt, "images": [out_file_name]}


def create_worker():
    """Factory function to create sample pipeline."""
    cfg = DeploymentEnv()

    pipeline = SampleWorker(
        num_gpus=cfg.num_gpus,
        checkpoint_dir=cfg.checkpoint_dir,
        model_name=cfg.model_name,
    )

    return pipeline
