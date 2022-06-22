import numpy as np
from counterfit.module import CFAlgo
from counterfit.targets import CFTarget

from art.estimators.classification.blackbox import BlackBoxClassifierNeuralNetwork
from art.utils import compute_success_array


class ArtEvasionAttack(CFAlgo):
    def build(
        self, target: CFTarget, channels_first: bool, clip_values: tuple
    ) -> object:
        """
        Build the attack.
        """

        # Build the blackbox classifier
        target_classifier = BlackBoxClassifierNeuralNetwork(
            target.predict_wrapper,
            target.input_shape,
            len(target.output_classes),
            channels_first=channels_first,
            clip_values=clip_values,
        )

        self.attack = self.attack_cls(target_classifier)
        return True

    def run(self, x, y=None, params=None):
        # Give the framework an opportunity to preprocess any thing in the attack.
        self.pre_attack_processing()
        results = self.attack.generate(x=np.array(x, dtype=np.float32), y=y)

        return results

    def pre_attack_processing(self):
        pass

    def post_attack_processing(self):
        pass

    def evasion_success(self, cfattack):
        if cfattack.options.__dict__.get("targeted") == True:
            labels = cfattack.options.target_labels
            targeted = True
        else:
            labels = cfattack.initial_labels
            targeted = False

        success = compute_success_array(
            cfattack.attack._estimator,
            cfattack.samples,
            labels,
            cfattack.results,
            targeted,
        )

        final_outputs, final_labels = cfattack.target.get_sample_labels(
            cfattack.results
        )

        cfattack.final_labels = final_labels
        cfattack.final_outputs = final_outputs
        return success

    def set_parameters(self) -> None:
        # ART has its own set_params function. Use it.
        attack_params = {}
        for k, v in self.options.attack_parameters.items():
            attack_params[k] = v["current"]
        self.attack.set_params(**attack_params)