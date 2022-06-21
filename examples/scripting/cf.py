import counterfit
from counterfit.targets import CFTarget
from transformers import AutoFeatureExtractor
from transformers import ResNetForImageClassification
from datasets import load_dataset
import torch


class MyTarget(CFTarget):
    def load(self):
        self.input_shape = (28, 28)
        self.output_classes = [0, 1]
        # Load the model
        dataset = load_dataset("huggingface/cats-image")

        image = dataset["test"]["image"][0]

        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.samples = [
            feature_extractor(image, return_tensors="pt").get("pixel_values")
        ]

        self.endpoint = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50"
        )

    def predict(self, x):
        with torch.no_grad():
            logits = self.endpoint(x).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        print(self.endpoint.config.id2label[predicted_label])
        return logits


# Test the load and predict
mt = MyTarget()
mt.load()
mt.predict(mt.samples[0])


from counterfit.modules.algo.art.evasion.hopskipjump import CFHopSkipJump
from IPython import embed


hsj = CFHopSkipJump()
embed()
