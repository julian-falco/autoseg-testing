import os
import sys
import json
import numpy as np
import time
import traceback
from itertools import product
from funlib.persistence import open_ds
from multiprocess import Pool

self_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
scripts_path = os.path.join(self_dir, "..")
os.chdir(scripts_path)

import hierarchical
import quick_eval

config = {
    "predict_fn_fp": "../02_train/setup_01/predict.py",  # 2
    "series_fp": ("../01_data/oblique.zarr", "../01_data/spine.zarr"),  # 2
    "predict_kwargs": {
        "iterations": (50000, 75000, 100000)  # 3
    },
    "segment_kwargs": {
        "normalize_preds": (False, True),  # 2
        "min_seed_distance": (5, 10, 15, 20),  # 4
        "boundary_mask": (False, True),  # 2
        "merge_function": hierarchical.waterz_merge_function.keys(),  # 11
        "erode_steps": (0, 1, 2),  # 3
        "clean_up": (0, 25, 50, 75, 100, 150, 200),  # 7
        "filter_value": (
            tuple(np.arange(0, 0.1, 0.01)) +
            tuple(np.arange(0.1, 1, 0.1))
        )  # 19
    }
}

# get the predict function

# predict_fn_dir = os.path.dirname(config["predict_fn_fp"])
# sys.path.append(predict_fn_dir)
# from predict import predict
# sys.path.remove(predict_fn_dir)

def run_predictions():
    # set up all kwarg combinations
    product_args = tuple([enumerate(args) for args in config["predict_kwargs"].values()])
    all_args = product(*product_args)
    for test_series in config["series_fp"]:
        raw_ds = "raw/s2"
        out_file = os.path.basename(test_series)
        for arg_set in all_args:
            # get the indexes
            indexes = tuple([arg[0] for arg in arg_set])
            # get the kwargs
            keys = tuple(config["predict_kwargs"].keys())
            values = tuple([arg[1] for arg in arg_set])
            kwargs = dict(zip(keys, values))
            # add custom kwargs
            kwargs["raw_ds"] = raw_ds
            kwargs["out_file"] = out_file
            # run predict
            predict(**kwargs)

def run_hierarchical():
    # set up the zarr grid
    shape = [2]  # custom: for the two predict functions
    shape += [len(args) for args in config["predict_kwargs"].values()]
    shape += [len(args) for args in config["segment_kwargs"].values()]
    shape += [20]  # for thresholds
    total_combos = np.product(shape)
    
    if not os.path.isdir("testing_log"):
        os.mkdir("testing_log")
    
    # set up product
    product_args = [enumerate(args) for args in config["predict_kwargs"].values()]
    product_args += [enumerate(args) for args in config["segment_kwargs"].values()]
    all_args = product(*tuple(product_args))
    keys = tuple(list(config["predict_kwargs"].keys()) + 
                 list(config["segment_kwargs"].keys()))
    
    # run the grid thing
    for series in config["series_fp"]: 
        group_name = os.path.basename(series)[:-5]  # remove .zarr extension
        if not os.path.isdir(os.path.join("testing_log", group_name)):
            os.mkdir(os.path.join("testing_log", group_name))
        pred_file = os.path.basename(series)
        labels = open_ds(
            series,
            "labels/s2"
        )
        try:
            mask = open_ds(
                series,
                "labels_mask/s2"
            )
        except:
            mask = None
        
        def send_it(arg_set):
            # get the indexes
            indexes = [setup_i] + [arg[0] for arg in arg_set]
            indexes_basename = "-".join(indexes) + ".json"
            # skip the test if already performed
            log_fp = os.path.join("testing_log", group_name, indexes_basename)
            if os.path.isdir(os.path.join(log_fp)):
                return
            
            start_time = time.time()
            
            # create the json dictionary
            log_data = {}
            
            # get the kwargs
            values = tuple([arg[1] for arg in arg_set])
            kwargs = dict(zip(keys, values))

            # get datasets and roi
            pred_ds = f"affs_{setup}_{kwargs['iterations']}"
            pred_roi = open_ds(pred_file, pred_ds).roi
            intersect_roi = labels.roi.intersect(pred_roi)
            
            # add custom kwargs
            kwargs["pred_file"] = pred_file
            kwargs["pred_dataset"] = pred_ds
            kwargs["roi"] = (tuple(intersect_roi.offset - pred_roi.offset), tuple(intersect_roi.shape))
            
            # output kwargs to user
            print(f"Setup {setup}")
            print(kwargs)
            print()
            
            # add kwargs to json file
            log_data["kwargs"] = kwargs.copy()

            # remove kwargs from predict function
            for k in config["predict_kwargs"]:
                del(kwargs[k])
            
            # run hierarchical
            log_data["thresholds"] = {}
            try:
                thresh_segs, fragments = hierarchical.post(**kwargs)
                for threshold_i, threshold in enumerate(sorted(thresh_segs.keys())):
                    seg = thresh_segs[threshold]
                    labels_arr = labels.to_ndarray(pred_roi, fill_value=0)
                    if mask is not None:
                        mask_arr = mask.to_ndarray(pred_roi, fill_value=0)
                    else:
                        mask_arr = None
                    metrics = quick_eval.evaluate(seg, labels_arr, mask_arr)
                    log_data["thresholds"][threshold] = metrics
                log_data["exception"] = ""
            
            except:
                log_data["exception"] = traceback.format_exc()
            
            finish_time = time.time()

            log_data["start_time"] = f"{start_time:.2f}"
            log_data["finish_time"] = f"{finish_time:.2f}"
            log_data["time_elapsed"] = f"{finish_time - start_time:.2f}"

            with open(log_fp, "w") as f:
                json.dump(log_data, f)
        
        for setup_i, setup in enumerate(("01", "02")):  # custom to two predict functions
            with Pool() as p:
                p.map(send_it, all_args)


if __name__ == "__main__":
    run_hierarchical()