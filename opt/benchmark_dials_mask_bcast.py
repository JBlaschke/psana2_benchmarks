#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from   argparse import ArgumentParser
from   pickle   import load, dump

from benchmarking import event_here, start, stop, log, event_log



@log
def bcast_dials_mask(comm, invalid_pixel_mask_path):

    if comm.Get_rank() == 0:

        # Check if environment variables are defined <-- I don't know what this
        # is good for, but the original xtc_process.py has it
        start("get_environ")
        PS_CALIB_DIR = os.environ.get('PS_CALIB_DIR')
        assert PS_CALIB_DIR
        stop("get_environ")

        # Load pickle -- this is different from CCTBX because there we're using
        # easy_pickle <-- I am not importing easy_pickle because I don't want
        # to enter dependency hell
        start("load_invalid_pixel_mask")
        with open(invalid_pixel_mask_path, "rb") as f:
            dials_mask = load(f)
        stop("load_invalid_pixel_mask")

    else:

        dials_mask = None

    # Bcast the dials_mask array
    start("bcast_dials_mask")
    dials_mask = comm.bcast(dials_mask, root=0)
    stop("bcast_dials_mask")

    return dials_mask



if __name__ == "__main__":

    # Defaul data
    default_parameters = {
        "invalid_pixel_mask_path" :
            "/img/run/psana-nersc/demo19/cxid9114/input/mask_ld91.pickle"
    }


    #
    # Parse input arguments
    #

    parser = ArgumentParser()
    parser.add_argument("--invalid_pixel_mask_path",
                        help="Path the the invalid pixel mask pickle file")
    parser.add_argument("--of",
                        help="Log dir -- every rank will write its own log file")

    # Get args dict, and sanitize None types
    args         = vars(parser.parse_args())

    output_name             = args["of"]
    invalid_pixel_mask_path = args["invalid_pixel_mask_path"]


    #
    # Initialize MPI
    #

    start("INIT MPI")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    stop("INIT MPI")

    rank = comm.Get_rank() # each process in MPI has a unique id, 0-indexed

    #
    # Run Benchmark
    #

    if rank == 0:
        print("MPI Initialize, Running bcast_dials_mask Benchmark")

    bcast_dials_mask(comm, invalid_pixel_mask_path)


    #
    # Save log files
    #

    if rank == 0:
        print("Writing logs")

    log_path = os.path.join(output_name, f"debug_{rank}.txt")
    with open(log_path, "w") as f:
        for entry in event_log(cctbx_fmt=True):
            print(entry, file=f)
