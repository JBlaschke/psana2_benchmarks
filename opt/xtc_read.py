#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import psana
from   argparse import ArgumentParser
from   pickle   import dump



def get_calib_data(ds):

    calib_data = list()
    for run in ds.runs():
        print(f"Getting Calibration for run: {run.runnum}")
        calib_data.append(run.calibconst)

    return calib_data



def set_defaults(args, defaults):

    new_args = dict()

    for key in args:
        if args[key] == None:
            if key in defaults.keys():
                new_args[key] = defaults[key]
        else:
            new_args[key] = args[key]

    return new_args



if __name__ == "__main__":

    # Defaul data
    default_parameters = {
        "exp"          : "cxid9114",
        "run"          : 1,
        "dir"          : "/img/data/xtc_test",
        "max_events"   : 0,
        "det_name"     : "cspad"
    }


    # Input args allowed by psana.DataSource
    psana_args = ["exp", "run", "dir", "max_events", "det_name", "batch_size"]


    #
    # Parse input arguments
    #

    parser = ArgumentParser()

    for arg in psana_args:
        parser.add_argument(f"--{arg}", help="psana.DataSource kwarg")

    parser.add_argument("--of", help="Name of output file")

    # Get args dict, and sanitize None types
    args         = vars(parser.parse_args())

    output_name  = args["of"]
    del args["of"]  # don't pass this to psana

    psana_kwargs = set_defaults(args, default_parameters)


    #
    # Load calib data for all runs
    #

    print("Collecting calibration data from psana.DataSource for:")
    print(f"{psana_kwargs}")

    ds         = psana.DataSource(**psana_kwargs)
    calib_data = get_calib_data(ds)


    #
    # Save calib data as pickle
    #

    print(f"Writing output to: {output_name}")

    with open(output_name, "wb") as f:
        dump(calib_data, f)
