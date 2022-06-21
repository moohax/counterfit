from abc import ABC
from abc import abstractmethod
import json

import numpy as np
from rich.table import Table


class CFReportGenerator(ABC):
    @abstractmethod
    def printable(target, batch):
        raise NotImplementedError()

    @abstractmethod
    def get_run_summary(target, cfattack):
        raise NotImplementedError()

    def printable_numpy(batch):
        o = np.get_printoptions()
        np.set_printoptions(
            threshold=30,
            precision=2,
            floatmode="maxprec_equal",
            formatter=dict(float=lambda x: f"{x:4.2f}"),
        )
        result = [str(np.array(row)).replace("\n", " ") for row in batch]
        np.set_printoptions(
            threshold=o["threshold"],
            precision=o["precision"],
            floatmode=o["floatmode"],
            formatter=o["formatter"],
        )
        return result

    def get_scan_summary(list_of_runs):
        # summarize by attack -- what is the best
        #   - success rate
        #   - average time
        #   - best result (highest confidence confusion)
        #   - attack_id for best parameters
        #   - attack_name

        total_successes = sum([s["successes"] for s in list_of_runs])
        total_runs = sum([s["batch_size"] for s in list_of_runs])
        times = [s["elapsed_time"] for s in list_of_runs]
        queries = [s["queries"] for s in list_of_runs]
        best_attack = None
        best_id = None
        best_score = None
        best_params = None
        best_queries = None
        for s in list_of_runs:
            for conf, il, fl in zip(
                s["final_confidence"], s["initial_label"], s["final_label"]
            ):
                if (s["targeted"] and s["target_class"] == fl) or (
                    s["targeted"] == False and fl != il
                ):
                    if (
                        best_score is None
                        or conf > best_score
                        or (conf == best_score and s["queries"] < best_queries)
                    ):
                        best_score = conf
                        best_id = s["attack_id"]
                        best_attack = s["attack_name"]
                        best_params = s["parameters"]
                        best_queries = s["queries"]
        return {
            "total_runs": total_runs,
            "total_successes": total_successes,
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "avg_queries": int(np.mean(queries)),
            "min_queries": np.min(queries),
            "max_queries": np.max(queries),
            "best_attack_name": best_attack,
            "best_attack_id": best_id,
            "best_attack_score": best_score,
            "best_params": best_params,
        }

    def printable_scan_summary(summaries_by_attack, summaries_by_label=None):
        """Print scan summaries in the command line console

        Args:
            summaries_by_attack ([dict]): Dictionary contains summary details per attack
            summaries_by_label ([dict], optional): Dictionary contains summary details per label. Defaults to None.
        """
        print("\n =============== \n <SCAN SUMMARY> \n ===============\n\n")
        table = Table(header_style="bold magenta")
        table.add_column("Attack Name")
        table.add_column("Total Runs")
        table.add_column("Successes (%)")
        table.add_column("Best Score (attack_id)")
        table.add_column("Best Parameters", width=110)

        for name, summary in summaries_by_attack.items():
            frac = summary["total_successes"] / summary["total_runs"]
            successes = f"{summary['total_successes']} ({frac:>.1%})"
            best = (
                f"{summary['best_attack_score']:0.1f} ({summary['best_attack_id']})"
                if summary["best_attack_score"]
                else "N/A"
            )
            best_params = json.dumps(summary["best_params"])
            table.add_row(
                str(name),
                str(summary["total_runs"]),
                str(successes),
                str(best),
                str(best_params),
            )
        st = Table(header_style="bold magenta")
        print(table)

        if summaries_by_label is not None:
            st.add_column("Class Label")
            st.add_column("Total Runs")
            st.add_column("Successes (%)")
            st.add_column("Best Score (Attack)")
            for name, summary in sorted(summaries_by_label.items()):
                frac = summary["total_successes"] / summary["total_runs"]
                successes = f"{summary['total_successes']} ({frac:>.1%})"
                best = (
                    f"{summary['best_attack_score']:0.1f} ({summary['best_attack_name']})"
                    if summary["best_attack_score"]
                    else "N/A"
                )
                st.add_row(
                    str(name), str(summary["total_runs"]), str(successes), str(best)
                )

            print(st)
        output = ""
        times_str = f"{summary['min_time']:>4.1f}/{summary['avg_time']:>4.1f}/{summary['max_time']:>4.1f}"
        queries_str = f"{summary['min_queries']:>5d}/{summary['avg_queries']:>5d}/{summary['max_queries']:>5d}"
        output += f"[+] Time[sec] (min/avg/max) {times_str} \n"
        output += f"\n[+] Queries (min/avg/max) {queries_str} \n"
        print(output)


import numpy as np
import hashlib


from counterfit.reporting import CFReportGenerator
from counterfit.utils import transform_numpy_to_bytes, get_predict_folder
from counterfit.data import ImageDataType
from PIL import Image

from rich.table import Table


class ImageReportGenerator(CFReportGenerator):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def printable(cls, target, batch, prefix=""):
        result = []
        i = 0
        for img_arr in batch:
            target_name = target.name
            _id = hashlib.md5(transform_numpy_to_bytes(img_arr)).hexdigest()[:8]
            basename = prefix + f"-{target_name}-{_id}-sample-{i}"
            predict_folder = get_predict_folder(target)
            filename = ImageReportGenerator.save(
                target, img_arr, results_path=predict_folder, filename=basename
            )
            result.append(filename)
            i += 1
        return result

    @classmethod
    def save(
        cls,
        target,
        array,
        suffix="",
        extension="png",
        results_path=None,
        filename=None,
        save_output=False,
    ):
        model_name = target.name
        if filename is None:
            filename = f"{model_name}-{target.target_id}"
        if suffix:
            filename += f"-{suffix}"
        filename += f".{extension}"
        array = np.array(array)
        array[np.isnan(array)] = 0  # change NaNs to 0s
        if np.max(array) > 1.0:
            array = ImageDataType.convert_to_uint8(array)
        elif np.max(array) <= 1.0:
            array = ImageDataType.convert_to_uint8(array, 255.0)
        else:
            raise ValueError(
                "Cannot determine image type from clip_values.  Expecting: (0,1) or (0,255)"
            )
        if len(target.input_shape) == 3:  # color channel?
            if not ImageDataType.is_channels_last(target.input_shape):
                # If channels are last. Convert to channels first
                array = array.transpose(1, 2, 0)
            # save mode is "L" or "RGB"
            save_mode = ImageDataType.get_channels(target.input_shape)

            im = Image.fromarray(array.squeeze(), mode=f"{save_mode}")

        elif len(target.input_shape) == 2:  # grayscale
            im = Image.fromarray(array, "L")

        else:
            raise ValueError("Expecting at least 2-dimensional image in input_shape")
        if results_path:
            filename = results_path + f"/{filename}"

        if save_output:
            im.save(filename)
        # elif target.target_task == "object_detection":
        #     indices, confidences, class_ids, boxes = target.get_indices_conf_u_nms(array, filter_enabled_class=True)
        #     ImageReportGenerator.save_image_w_bbs(array, indices, confidences, class_ids, boxes, target.final_output_classes, filename)
        # else:
        #     raise ValueError(f"{target.target_task} {target.target_data_type} save not supported at this time...")
        return filename

    @classmethod
    def get_run_summary(cls, cfattack):
        # count successes
        success_indicator = cfattack.success
        batch_size = len(success_indicator)
        successes = sum(success_indicator)

        # initial scores/labels
        i_0 = np.atleast_2d(cfattack.samples)
        o_0 = np.atleast_2d(cfattack.initial_outputs)
        l_0 = np.array(cfattack.initial_labels)

        # final scores/labels
        if cfattack.results is None:
            # failed attack? adopt originals
            i_f, o_f, l_f = i_0, o_0, l_0
        else:
            i_f = np.array(cfattack.results)
            o_f = np.atleast_2d(cfattack.final_outputs)
            l_f = np.array(cfattack.final_labels)

        # l2 norm
        i_0 = i_0.reshape(batch_size, -1).astype(float)
        i_f = i_f.reshape(batch_size, -1).astype(float)
        metric = "Max Abs Chg."

        max_abs_change = np.atleast_1d(abs(i_f - i_0).max(axis=-1))

        conf_0 = np.array(
            [
                o_0[i][cfattack.target.output_classes.index(lab)]
                for i, lab in enumerate(l_0)
            ]
        )

        conf_f = np.array(
            [
                o_f[i][cfattack.target.output_classes.index(lab)]
                for i, lab in enumerate(l_f)
            ]
        )

        degenerate = np.logical_and(l_0 != l_f, success_indicator == True)
        run_summary = {
            "batch_size": batch_size,
            "successes": successes,
            "input_change": max_abs_change,
            "input_change_metric": metric,
            "initial_confidence": conf_0,
            "final_confidence": conf_f,
            "initial_label": l_0,
            "final_label": l_f,
            "sample_index": np.atleast_1d(
                cfattack.options.cf_options["sample_index"]["current"]
            ),
            "type": cfattack.target.data_type,
            # 'result': result,
            "elapsed_time": cfattack.elapsed_time,
            "queries": cfattack.logger.num_queries,
            "attack_name": cfattack.name,
            "attack_id": cfattack.attack_id,
            "parameters": cfattack.options.attack_parameters,
            "targeted": cfattack.options.attack_parameters["targeted"]
            if "targeted" in cfattack.options.attack_parameters.keys()
            else False,
            "target_label": cfattack.options.attack_parameters["target_labels"]
            if "targeted" in cfattack.options.attack_parameters.keys()
            else "",
            "degenerate": degenerate,
        }
        return run_summary

    @classmethod
    def print_run_summary(cls, summary):
        stats_table = Table(header_style="bold magenta")
        stats_table.add_column("Success", no_wrap=True)
        stats_table.add_column("Elapsed time", no_wrap=True)
        stats_table.add_column("Total Queries", no_wrap=True)

        if summary["elapsed_time"] > summary["queries"]:
            query_rate = summary["elapsed_time"] / summary["queries"]
            units = "sec/query"
        else:
            query_rate = summary["queries"] / summary["elapsed_time"]
            units = "query/sec"

        stats_table.add_row(
            f"{summary['successes']}/{summary['batch_size']}",
            f"{summary['elapsed_time']:.1f}",
            f"{summary['queries']:.0f} ({query_rate:.1f} {units})",
        )

        print(stats_table)

        table = Table(header_style="bold magenta")
        metric = summary["input_change_metric"]
        table.add_column("Sample Index")
        table.add_column("Input Label (conf)")
        table.add_column("Adversarial Label (conf)")
        table.add_column(f"{metric}")
        table.add_column("Adversarial Input", width=110)
        table.add_column("success")
        for i, (si, li, conf_0, lf, conf_f, change, res, d) in enumerate(
            zip(
                summary["sample_index"],
                summary["initial_label"],
                summary["initial_confidence"],
                summary["final_label"],
                summary["final_confidence"],
                summary["input_change"],
                summary["result"],
                summary["degenerate"],
            )
        ):
            label_confidence = f"{li} ({conf_0:.4f})"
            final_confidence = f"{lf} ({conf_f:.4f})"
            table.add_row(
                str(si),
                str(label_confidence),
                str(final_confidence),
                str(round(change, 4)),
                str(res),
                str(d),
            )

        print(table)


import numpy as np
from counterfit.reporting import CFReportGenerator

from rich.table import Table


class TabularReportGenerator(CFReportGenerator):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def printable(cls, target, batch, prefix=""):
        o = np.get_printoptions()
        np.set_printoptions(
            threshold=30,
            precision=2,
            floatmode="maxprec_equal",
            formatter=dict(float=lambda x: f"{x:4.2f}"),
        )
        result = [str(np.array(row)).replace("\n", " ") for row in batch]
        np.set_printoptions(
            threshold=o["threshold"],
            precision=o["precision"],
            floatmode=o["floatmode"],
            formatter=o["formatter"],
        )
        return result

    @classmethod
    def get_run_summary(cls, cfattack):
        # count successes
        success_indicator = np.array(cfattack.success)  # numpy bool array
        batch_size = len(success_indicator)
        successes = sum(success_indicator)

        # initial scores/labels
        i_0 = cfattack.samples
        o_0 = cfattack.initial_outputs
        l_0 = cfattack.initial_labels

        # final scores/labels
        if cfattack.results is None:
            # failed attack? adopt originals
            i_f, o_f, l_f = i_0, o_0, l_0
        else:
            i_f = cfattack.results
            o_f = cfattack.final_outputs
            l_f = cfattack.final_labels

        # l2 norm
        i_0 = i_0.reshape(batch_size, -1).astype(float)
        i_f = i_f.reshape(batch_size, -1).astype(float)
        metric = "% Eucl. dist."
        eps = np.finfo("float32").eps
        rel_distance = np.sqrt(np.nansum(np.square(i_f - i_0), axis=1)) / (
            np.linalg.norm(i_0, axis=1) + eps
        )

        result = TabularReportGenerator.printable(cfattack.target, i_f, prefix=None)

        conf_0 = np.array(
            [
                o_0[i][cfattack.target.output_classes.index(lab)]
                for i, lab in enumerate(l_0)
            ]
        )
        conf_f = np.array(
            [
                o_f[i][cfattack.target.output_classes.index(lab)]
                for i, lab in enumerate(l_f)
            ]
        )

        degenerate = np.logical_and(l_0 != l_f, success_indicator == True)

        run_summary = {
            "batch_size": batch_size,
            "successes": successes,
            "input_change": rel_distance,
            "input_change_metric": metric,
            "initial_confidence": conf_0,
            "final_confidence": conf_f,
            "initial_label": l_0,
            "final_label": l_f,
            "sample_index": np.atleast_1d(
                cfattack.options.cf_options["sample_index"]["current"]
            ),
            "type": cfattack.target.data_type,
            "result": result,
            "elapsed_time": cfattack.elapsed_time,
            "queries": cfattack.logger.num_queries,
            "attack_name": cfattack.name,
            "attack_id": cfattack.attack_id,
            "parameters": cfattack.options.attack_parameters,
            "targeted": cfattack.options.attack_parameters["targeted"]
            if "targeted" in cfattack.options.attack_parameters.keys()
            else False,
            "target_label": cfattack.options.attack_parameters["target_labels"]
            if "targeted" in cfattack.options.attack_parameters.keys()
            else "",
            "degenerate": degenerate,
        }
        return run_summary

    @classmethod
    def print_run_summary(cls, summary):
        stats_table = Table(header_style="bold magenta")
        stats_table.add_column("Success", no_wrap=True)
        stats_table.add_column("Elapsed time", no_wrap=True)
        stats_table.add_column("Total Queries", no_wrap=True)

        if summary["elapsed_time"] > summary["queries"]:
            query_rate = summary["elapsed_time"] / summary["queries"]
            units = "sec/query"
        else:
            query_rate = summary["queries"] / summary["elapsed_time"]
            units = "query/sec"

        stats_table.add_row(
            f"{summary['successes']}/{summary['batch_size']}",
            f"{summary['elapsed_time']:.1f}",
            f"{summary['queries']:.0f} ({query_rate:.1f} {units})",
        )

        print(stats_table)

        table = Table(header_style="bold magenta")
        metric = summary["input_change_metric"]
        table.add_column("Sample Index")
        table.add_column("Input Label (conf)")
        table.add_column("Adversarial Label (conf)")
        table.add_column(f"{metric}")
        table.add_column("Adversarial Input", width=110)
        table.add_column("success")
        for i, (si, li, conf_0, lf, conf_f, change, res, d) in enumerate(
            zip(
                summary["sample_index"],
                summary["initial_label"],
                summary["initial_confidence"],
                summary["final_label"],
                summary["final_confidence"],
                summary["input_change"],
                summary["result"],
                summary["degenerate"],
            )
        ):
            label_confidence = f"{li} ({conf_0:.4f})"
            final_confidence = f"{lf} ({conf_f:.4f})"
            table.add_row(
                str(si),
                str(label_confidence),
                str(final_confidence),
                str(round(change, 4)),
                str(res),
                str(d),
            )

        print(table)


from counterfit.reporting import CFReportGenerator
from rich.table import Table
import numpy as np


class TextReportGenerator(CFReportGenerator):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def printable(target, batch, prefix=""):
        return batch

    @classmethod
    def get_run_summary(cls, cfattack):
        # count successes
        success_indicator = cfattack.success
        batch_size = len(success_indicator)
        successes = sum(success_indicator)

        # initial scores/labels
        i_0 = cfattack.samples
        o_0 = cfattack.initial_outputs
        l_0 = cfattack.initial_labels

        # final scores/labels
        if cfattack.results is None:
            # failed attack? adopt originals
            i_f, o_f, l_f = i_0, o_0, l_0
        else:
            i_f = cfattack.results
            o_f = cfattack.final_outputs
            l_f = cfattack.final_labels

        metric = "% edit dist."
        distances = [Levenshtein.distance(iif, ii0) for iif, ii0 in zip(i_f, i_0)]
        rel_distance = [d / len(ii0) for d, ii0 in zip(distances, i_0)]
        result = TextReportGenerator.printable(cfattack.target, i_f)

        conf_0 = []
        for i, lab in enumerate(l_0):
            conf_0.append(o_0[i][cfattack.target.output_classes.index(lab)])
        conf_0 = np.array(conf_0)

        conf_f = []
        for i, lab in enumerate(l_f):
            conf_f.append(o_f[i][cfattack.target.output_classes.index(lab)])
        conf_f = np.array(conf_f)

        degenerate = np.logical_and(l_0 != l_f, success_indicator == True)
        run_summary = {
            "batch_size": batch_size,
            "successes": successes,
            "input_change": rel_distance,
            "input_change_metric": metric,
            "initial_confidence": conf_0,
            "final_confidence": conf_f,
            "initial_label": l_0,
            "final_label": l_f,
            "sample_index": np.atleast_1d(
                cfattack.options.cf_options["sample_index"]["current"]
            ),
            "type": cfattack.target.data_type,
            "result": result,
            "elapsed_time": cfattack.elapsed_time,
            "queries": cfattack.logger.num_queries,
            "attack_name": cfattack.name,
            "attack_id": cfattack.attack_id,
            "parameters": cfattack.options.attack_parameters,
            "targeted": cfattack.options.attack_parameters["targeted"]
            if "targeted" in cfattack.options.attack_parameters.keys()
            else False,
            "target_label": cfattack.options.attack_parameters["target_labels"]
            if "targeted" in cfattack.options.attack_parameters.keys()
            else "",
            "degenerate": degenerate,
        }
        return run_summary

    @staticmethod
    def print_run_summary(summary):
        stats_table = Table(header_style="bold magenta")
        stats_table.add_column("Success", no_wrap=True)
        stats_table.add_column("Elapsed time", no_wrap=True)
        stats_table.add_column("Total Queries", no_wrap=True)

        if summary["elapsed_time"] > summary["queries"]:
            query_rate = summary["elapsed_time"] / summary["queries"]
            units = "sec/query"
        else:
            query_rate = summary["queries"] / summary["elapsed_time"]
            units = "query/sec"

        stats_table.add_row(
            f"{summary['successes']}/{summary['batch_size']}",
            f"{summary['elapsed_time']:.1f}",
            f"{summary['queries']:.0f} ({query_rate:.1f} {units})",
        )

        print(stats_table)

        table = Table(header_style="bold magenta")
        metric = summary["input_change_metric"]
        table.add_column("Sample Index")
        table.add_column("Input Label (conf)")
        table.add_column("Adversarial Label (conf)")
        table.add_column(f"{metric}")
        table.add_column("Adversarial Input", width=110)
        table.add_column("success")
        for i, (si, li, conf_0, lf, conf_f, change, res, d) in enumerate(
            zip(
                summary["sample_index"],
                summary["initial_label"],
                summary["initial_confidence"],
                summary["final_label"],
                summary["final_confidence"],
                summary["input_change"],
                summary["result"],
                summary["degenerate"],
            )
        ):
            label_confidence = f"{li} ({conf_0:.4f})"
            final_confidence = f"{lf} ({conf_f:.4f})"
            table.add_row(
                str(si),
                str(label_confidence),
                str(final_confidence),
                # str(input_change),
                str(round(change, 4)),
                str(res),
                str(d),
            )

        print(table)