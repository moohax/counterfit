import numpy as np
import importlib
import inspect
import re

from counterfit.module import CFModule
from counterfit.targets import CFTarget

from art.utils import compute_success_array, random_targets
from scipy.stats import entropy
from art.utils import clip_and_round


from art.estimators.classification.blackbox import BlackBoxClassifierNeuralNetwork


class ArtEvasionAttack(CFModule):
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

    @classmethod
    def run(cls, cfattack):

        # Give the framework an opportunity to preprocess any thing in the attack.
        cls.pre_attack_processing(cfattack)

        # Find the appropriate "run" function
        attack_attributes = cfattack.attack.__dir__()

        # Run the attack. Each attack type has it's own execution function signature.
        if "infer" in attack_attributes:
            x_init_average = np.zeros((10, 1, 784)) + np.mean(cfattack.target.X, axis=0)
            results = cfattack.attack.infer(
                x_init_average,
                np.array(cfattack.target.output_classes).astype(np.int64),
            )
            # results = cfattack.attack.infer(cfattack.samples, np.array(cfattack.target.output_classes).astype(np.int64))

        elif "reconstruct" in attack_attributes:
            results = cfattack.attack.reconstruct(
                np.array(cfattack.samples, dtype=np.float32)
            )

        elif "generate" in attack_attributes:

            if "ZooAttack" == cfattack.name:
                # patch ZooAttack
                cfattack.attack.estimator.channels_first = True

            results = cfattack.attack.generate(
                x=np.array(cfattack.samples, dtype=np.float32)
            )

        elif "poison" in attack_attributes:
            results = cfattack.attack.poison(
                np.array(cfattack.samples, dtype=np.float32)
            )

        elif "poison_estimator" in attack_attributes:
            results = cfattack.attack.poison(
                np.array(cfattack.samples, dtype=np.float32)
            )

        elif "extract" in attack_attributes:
            # Returns a thieved classifier
            training_shape = (len(cfattack.target.X), *cfattack.target.input_shape)

            samples_to_query = cfattack.target.X.reshape(training_shape).astype(
                np.float32
            )
            results = cfattack.attack.extract(
                x=samples_to_query, thieved_classifier=cfattack.attack.estimator
            )

            cfattack.thieved_classifier = results
        else:
            print("Not found!")
        return results

    @classmethod
    def pre_attack_processing(cls, cfattack):
        cls.set_parameters(cfattack)

    @staticmethod
    def post_attack_processing(cfattack):
        attack_attributes = cfattack.attack.__dir__()

        pass

        # if "generate" in attack_attributes:
        #     current_datatype = cfattack.target.data_type
        #     current_dt_report_gen = get_target_data_type_obj(current_datatype)
        #     cfattack.summary = current_dt_report_gen.get_run_summary(cfattack)
        #     # current_dt_report_gen.print_run_summary(summary)

        # elif "extract" in attack_attributes:
        #     # Override default reporting for the attack type
        #     extract_table = Table(header_style="bold magenta")
        #     # Add columns to extraction table
        #     extract_table.add_column("Success")
        #     extract_table.add_column("Copy Cat Accuracy")
        #     extract_table.add_column("Elapsed time")
        #     extract_table.add_column("Total Queries")

        #     # Add data to extraction table
        #     success = cfattack.success[0]  # Starting value
        #     thieved_accuracy = cfattack.results
        #     elapsed_time = cfattack.elapsed_time
        #     num_queries = cfattack.logger.num_queries
        #     extract_table.add_row(str(success), str(
        #         thieved_accuracy), str(elapsed_time), str(num_queries))

        #     print(extract_table)

    @classmethod
    def set_classifier(cls, target: CFTarget):

        # Match the target.classifier attribute with an ART classifier type
        classifiers = cls.get_classifiers()

        # If no classifer attribute has been set, assume a blackbox.
        if not hasattr(target, "classifier"):

            # If the target model returns log_probs, return the relevant estimator
            if hasattr(target, "log_probs"):
                if target.log_probs == True:
                    return classifiers.get("QueryEfficientGradientEstimationClassifier")

            # Else return a plain BB estimator
            else:
                return classifiers.get("BlackBoxClassifierNeuralNetwork")

        # Else resolve the correct classifier
        else:
            for classifier in classifiers.keys():
                if target.classifier.lower() in classifier.lower():
                    return classifiers.get(classifier, None)

    def check_success(self, cfattack) -> bool:
        attack_attributes = set(cfattack.attack.__dir__())

        if "generate" in attack_attributes:
            return self.evasion_success(cfattack)

        elif "extract" in attack_attributes:
            return self.extraction_success(cfattack)

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

    def extraction_success(self, cfattack):
        training_shape = (len(cfattack.target.X), *cfattack.target.input_shape)
        training_data = cfattack.target.X.reshape(training_shape)

        victim_preds = np.atleast_1d(
            np.argmax(cfattack.target.predict_wrapper(x=training_data), axis=1)
        )
        thieved_preds = np.atleast_1d(
            np.argmax(cfattack.thieved_classifier.predict(x=training_data), axis=1)
        )

        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        cfattack.results = acc

        if acc > 0.1:  # TODO add to options struct
            return [True]
        else:
            return [False]

    def set_parameters(self) -> None:
        # ART has its own set_params function. Use it.
        attack_params = {}
        for k, v in self.options.attack_parameters.items():
            attack_params[k] = v["current"]
        self.attack.set_params(**attack_params)