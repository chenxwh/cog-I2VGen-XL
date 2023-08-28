# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

# import os
# os.environ["MODELSCOPE_CACHE"] = "./model_cache_dir"

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.pipe_2iv = pipeline(
            task="image-to-video",
            model="damo/Image-to-Video",
            model_revision="v1.1.0",
            device="cuda:0",
        )
        self.pipe_v2v = pipeline(
            task="video-to-video",
            model="damo/Video-to-Video",
            model_revision="v1.1.0",
            device="cuda:0",
        )

    def predict(
        self,
        task: str = Input(
            description="Choose a task",
            choices=["image-to-video", "video-to-video"],
            default="image-to-video",
        ),
        input_file: Path = Input(description="Input image or video"),
        text: str = Input(
            description="Video description for video-to-video task.", default=None
        ),
        high_resolution: bool = Input(
            description="Option for image-to-video task, set to True to generate high-resolution video",
            default=False,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        output_i2v_path = "/tmp/i2v_output.mp4"
        output_v2v_path = "/tmp/v2v_output.mp4"

        if task == "image-to-video":
            # image to video
            self.pipe_2iv(str(input_file), output_video=output_i2v_path)[
                OutputKeys.OUTPUT_VIDEO
            ]
            if not high_resolution:
                return Path(output_i2v_path)

            # video resolution
            p_input = {"video_path": output_i2v_path}
            self.pipe_v2v(p_input, output_video=output_v2v_path)[
                OutputKeys.OUTPUT_VIDEO
            ]

            return Path(output_v2v_path)

        assert text, "Please provide text description for video-to-video task."
        p_input = {"video_path": str(input_file), "text": text}
        self.pipe_v2v(p_input, output_video=output_v2v_path)[OutputKeys.OUTPUT_VIDEO]
        return Path(output_v2v_path)
