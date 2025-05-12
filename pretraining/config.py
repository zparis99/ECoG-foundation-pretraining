from dataclasses import dataclass, field
from typing import Optional

from ecog_foundation_model.config import VideoMAETaskConfig, VideoMAEExperimentConfig


@dataclass
class PretrainingTaskConfig(VideoMAETaskConfig):
    model_name: Optional[str] = None


@dataclass
class PretrainingExperimentConfig(VideoMAEExperimentConfig):
    video_mae_task_config: PretrainingTaskConfig = field(
        default_factory=VideoMAETaskConfig
    )
