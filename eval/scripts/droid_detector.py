import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)


class DroidCocoDetector:
    def __init__(self, device="cuda:0", confidence_threshold=0.55):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold

        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.categories = weights.meta["categories"]

        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        self.model.transform.min_size = (480,)
        self.model.transform.max_size = 640
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def detect(self, frame):
        frame = np.asarray(frame)

        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB HWC frame, got {frame.shape}")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        image = (
            torch.from_numpy(np.ascontiguousarray(frame))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(self.device)
        )

        output = self.model([image])[0]

        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]

        selected = torch.where(scores >= self.confidence_threshold)[0]
        selected = selected[
            torch.argsort(scores[selected], descending=True)[:20]
        ]

        detections = []
        for index in selected.tolist():
            class_id = int(labels[index])
            detections.append({
                "box_xyxy": tuple(float(x) for x in boxes[index].tolist()),
                "confidence": float(scores[index]),
                "label": self.categories[class_id],
            })

        return detections


def factory(device="cpu"):
    return DroidCocoDetector(device=device)
