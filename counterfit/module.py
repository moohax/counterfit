from abc import abstractmethod


import os

from counterfit.utils import set_id


class CFModule:
    """Base class for all frameworks."""

    @abstractmethod
    def build(self, target, attack):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def pre_attack_processing(self, cfattack):
        pass

    @abstractmethod
    def post_attack_processing(self, cfattack):
        pass

    @abstractmethod
    def set_parameters(self, cfattack):
        pass


class CFAlgo(CFModule):
    """
    The base class for all algorithmic modules.
    """

    def __init__(self, name, target, framework, attack, options, scan_id=None):

        # Parent framework
        self.name = name
        self.attack_id = set_id()
        self.scan_id = scan_id
        self.target = target
        self.framework = framework
        self.attack = attack
        self.options = options

        # Attack information
        self.created_on = datetime.datetime.utcnow().strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        self.attack_status = "pending"

        # Algo parameters
        self.samples = None
        self.initial_labels = None
        self.initial_outputs = None

        # Attack results
        self.final_outputs = None
        self.final_labels = None
        self.results = None
        self.success = None
        self.elapsed_time = None

        # reporting
        self.run_summary = None

        # Get the samples.
        self.samples = target.get_samples(
            self.options.cf_options["sample_index"]["current"]
        )

        self.logger = self.set_logger(
            logger=self.options.cf_options["logger"]["current"]
        )
        self.target.logger = self.logger

    def prepare_attack(self):
        # Send a request to the target for the selected sample
        self.initial_outputs, self.initial_labels = self.target.get_sample_labels(
            self.samples
        )

    def set_results(self, results: object) -> None:
        self.results = results

    def set_status(self, status: str) -> None:
        self.attack_status = status

    def set_success(self, success: bool = False) -> None:
        self.success = success

    def set_logger(self, logger):
        new_logger = get_attack_logger_obj(logger)
        logger = new_logger()
        return logger

    def set_elapsed_time(self, start_time, end_time):
        self.elapsed_time = end_time - start_time

    def get_results_folder(self):
        results_folder = self.target.get_results_folder()

        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        scan_folder = os.path.join(results_folder, self.attack_id)
        if not os.path.exists(scan_folder):
            os.mkdir(scan_folder)

        return scan_folder

    def save_run_summary(self, filename=None, verbose=False):
        run_summary = {
            "sample_index": self.options.sample_index,
            "initial_labels": self.initial_labels,
            "final_labels": self.final_labels,
            "elapsed_time": self.elapsed_time,
            "num_queries": self.logger.num_queries,
            "success": self.success,
            "results": self.results,
        }

        if verbose:
            run_summary["input_samples"] = self.samples

        if not filename:
            results_folder = self.get_results_folder()
            filename = f"{results_folder}/run_summary.json"

        # with open(filename, "w") as summary_file:
        #     summary_file.write(data.decode())


# def build_algo(target, attack, scan_id=None):
#     """Build a new CFAttack.

#     Search through the loaded frameworks for the attack and create a new CFAttack object for use.

#     Args:
#         target_name (CFTarget, required): The target object.
#         attack_name (str, required): The attack name.
#         scan_id (str, Optional): A unique value

#     Returns:
#         CFAttack: A new CFAttack object.
#     """

#     # Resolve the attack
#     try:
#         for k, v in framework().items():
#             if attack in list(v["attacks"].keys()):
#                 framework = v["module"]()
#                 attack = v["attacks"][attack]

#     except Exception as error:
#         print(f"Failed to load framework or resolve {attack}: {error}")
#         traceback.print_exc()

#     # Ensure the attack is compatible with the target
#     if target.data_type not in attack["attack_data_tags"]:
#         print(
#             f"Target data type ({target.data_type}) is not compatible with the attack chosen ({attack['attack_data_tags']})"
#         )
#         return False

#     if hasattr(target, "classifier"):
#         print("Target classifier may not be compatible with this attack.")
#     else:
#         print(
#             "Target does not have classifier attribute set. Counterfit will treat the target as a blackbox."
#         )

#     # Have the framework build the attack.
#     try:
#         new_attack = framework.build(
#             target=target,
#             attack=attack["attack_class"],  # The dotted path of the attack.
#         )

#     except Exception as error:
#         print(f"Framework failed to build attack: {error}")
#         traceback.print_exc()

#     # Create a CFAttack object
#     try:
#         cfattack = CFModule(
#             name=attack["attack_class"],
#             target=target,
#             framework=framework,
#             attack=new_attack,
#             options=set_options(attack["attack_parameters"]),
#         )

#     except Exception as error:
#         print(f"Failed to build CFAttack: {error}")
#         traceback.print_exc()

#     return cfattack


# def _frameworks():
#     pass


# def _attacks():
#     pass


# import numpy as np


# def run_algo(self) -> bool:
#     """Run a prepared attack. Get the appropriate framework and execute the attack.

#     Args:
#         attack_id (str, required): The attack id to run.

#     Returns:
#         Attack: A new Attack object with an updated cfattack_class. Additional properties set in this function include, attack_id (str)
#         and the parent framework (str). The framework string is added to prevent the duplication of code in run_attack.
#     """

#     # Set the initial values for the attack. Samples, logger, etc.
#     self.prepare_attack()

#     # Run the attack
#     self.set_status("running")

#     # Start timing the attack for the elapsed_time metric
#     start_time = time.time()

#     # Run the attack
#     try:
#         results = self.framework.run(cfattack)
#     except Exception as error:
#         # postprocessing steps for failed attacks
#         success = [False] * len(self.initial_labels)

#         print(f"Failed to run {self.attack_id} ({self.name}): {error}")

#         results = None
#         return

#     # postprocessing steps for successful attacks
#     finally:
#         # Stop the timer
#         end_time = time.time()

#         # Set the elapsed time metric
#         self.set_elapsed_time(start_time, end_time)

#         # Set the results the attack returns
#         # Results are attack and framework specific.
#         self.set_results(results)

#         # Determine the success of the attack
#         success = self.framework.check_success(cfattack)

#         # Set the success value
#         self.set_success(success)

#         # Give the framework an opportunity to process the results, generate reports, etc
#         self.framework.post_attack_processing(cfattack)

#         # Mark the attack as complete
#         self.set_status("complete")

#         # Let the user know the attack has completed successfully.
#         print("Attack completed {}".format(self.attack_id))
