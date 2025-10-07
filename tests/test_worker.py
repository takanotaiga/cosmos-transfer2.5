from cosmos_transfer2.control2world import (
    Control2World_Inference,
    Control2World_Params,
)
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_gradio.deployment_env import DeploymentEnv
import json


def read_params(inference_json_file):
    with open(inference_json_file, "r") as f:
        params = json.load(f)
    return params


NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."


v1 = "data_local/assets/robot_example/vis/robot_vis_spec.json"
d1 = "data_local/assets/robot_example/depth/robot_depth_spec.json"
e1 = "data_local/assets/robot_example/edge/robot_edge_spec.json"
s1 = "data_local/assets/robot_example/seg/robot_seg_spec.json"
t2 = "data_local/assets/robot_example/robot_multi_modal_on_the_fly_spec.json"

# NOTE that the hint key has to be set at init time!!!!
sample_params_dict = {
    "video_path": "data_local/assets/robot_example/robot_input.mp4",
    "prompt_path": "data_local/assets/robot_example/robot_prompt.txt",
    "negative_prompt": NEGATIVE_PROMPT,
    "output_dir": "outputs/test_worker/",
    "edge": {
        "control_path": "data_local/assets/robot_example/edge/robot_edge.mp4",
    },
}


def test_transfer(model_name, params):
    cfg = DeploymentEnv()
    params = Control2World_Params.create(params)
    log.info(f"params: {json.dumps(params.to_kwargs(), indent=4)}")
    pipeline = Control2World_Inference(num_gpus=1, checkpoint_dir=cfg.checkpoint_dir, hint_key=model_name)

    log.info("Inference start****************************************")
    params.output_dir = f"outputs/test_worker/{model_name}"
    pipeline.infer(params.to_kwargs())
    log.info("Inference complete****************************************")


# Note that multiview requires 8 GPUs and cannot be tested w/o torchrun
if __name__ == "__main__":
    test_transfer("edge", sample_params_dict)
    # test_transfer("depth", read_params(d1))
    # test_transfer("edge", read_params(e1))
    # test_transfer("seg", read_params(s1))
