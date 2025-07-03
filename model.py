# model.py 

from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel
)
from ultralytics.utils import yaml_load, ROOT


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains 'world'."""
        stem = Path(model).stem  # filename without extension
        if "world" in stem:
            new_instance = YOLOWorld(model)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map task type to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }

    def set_classes(self, classes):
        """
        Set custom classes.
        Args:
            classes (List[str]): List of class names, e.g., ["pest", "insect", "larva"]
        """
        self.model.set_classes(classes)
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes
        if self.predictor:
            self.predictor.model.names = classes


class YOLOWorld(Model):
    """YOLO-World object detection model for open-vocabulary detection."""

    def __init__(self, model="yolov8s-world.pt") -> None:
        """
        Initialize YOLOWorld with the specified model file.
        Args:
            model (str): Path to YOLOWorld model, defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model, task="detect")
        self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map detect task to WorldModel, validator, and predictor."""
        return {
            "detect": {
                "model": yolo.world.WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
