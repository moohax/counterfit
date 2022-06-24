from hashlib import new
from PIL import Image
from counterfit.targets import CFTarget
from transformers import AutoFeatureExtractor
from transformers import ResNetForImageClassification
from datasets import load_dataset
import torch
import io
import numpy as np

from counterfit.modules.algo.art.evasion.hopskipjump import CFHopSkipJump


class MyTarget(CFTarget):
    def __init__(self):
        # self.input_shape = (640, 480, 3)
        self.output_classes = [i for i in range(0, 1000)]

        # Load the model
        dataset = load_dataset("huggingface/cats-image")

        self.image = np.array(dataset["test"]["image"][0]).astype(np.float32)

        self.input_shape = self.image.shape

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/resnet-50"
        )

        self.endpoint = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50"
        )

    def predict(self, x):

        # Convert to x to images so we can use the feature extractor.
        imgs = []
        for i in x:
            new_img = Image.fromarray(np.array(i).astype(np.uint8), "RGB")

            # Rebuild the batch from images.
            new_x = (
                self.feature_extractor(new_img, return_tensors="pt")
                .get("pixel_values")
                .numpy()
            )[0]

            imgs.append(new_x)

        # Keep this because I don't want to lose it.
        # membuf = io.BytesIO()
        # img.save(membuf, format="png")

        try:
            x = torch.from_numpy(np.array(imgs))
        except:
            pass

        with torch.no_grad():
            logits = self.endpoint(x).logits

        return logits.numpy()


if __name__ == "__main__":
    # Test the load and predict
    target = MyTarget()
    target.predict(target.image.reshape(1, *target.input_shape))

    hsj = CFHopSkipJump()
    hsj.run(target, target.image.reshape(1, *target.input_shape))

    # Image.fromarray((im_a * 255).astype(np.uint8), "RGB").save("./maybe.png")
