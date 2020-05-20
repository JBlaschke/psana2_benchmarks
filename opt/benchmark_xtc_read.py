#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import socket
import psana
import numpy    as np
from   argparse import ArgumentParser
from   pickle   import dump

from benchmarking import Event,\
                         set_defaults,\
                         event_here, start, stop, log, event_log




#
# PSANA2 BENCHMARK, based on CCTBX's XTC_PROCESS pipeline.
# COMMENT:  I've started with cctbx_project/xfel/xtc_process.py and stripped
# out all the things that I don't think are relevant to this benchmark
#




@log
def get_calib_file_path(env, address, run):
    """ Findes the path to the SLAC metrology file stored in a psana
    environment object's calibration store
    @param env psana environment object
    @param address address string for a detector
    @param run psana run object or run number
    """

    from psana import Detector


    #
    # try to get it from the detector interface
    #

    try:
        start("load geometry from detector")
        psana_det = Detector(address, run.env())
        ret       = psana_det.pyda.geoaccess(run.run()).path
        stop("load geometry from detector")

        return ret
    except Exception as e:
        pass


    #
    # try to get it from the calib store directly
    #

    from psana import ndarray_uint8_1, Source

    start("load geometry from calib store")
    cls      = env.calibStore()
    src      = Source('DetInfo(%s)'%address)
    path_nda = cls.get(ndarray_uint8_1, src, 'geometry-calib')
    stop("load geometry from calib store")

    if path_nda is None:
        return None
    return ''.join(map(chr, path_nda))



@log
def env_dxtbx_from_slac_metrology(run, address):
    """ Loads a dxtbx cspad cbf header only object from the metrology path
    stored in a psana run object's calibration store
    @param env psana run object
    @param address address string for a detector
    """

    start("load geometry data from detector")
    det      = run.Detector(address)
    geometry = det.raw.geometry()
    stop("load geometry data from detector")

    if geometry is None:
        metro_path = get_calib_file_path(run.env(), address, run)
    elif geometry.valid:
        metro_path = None
    else:
        raise RuntimeError(f"Could not read geometry, hostname: {socket.gethostname()}")

    if metro_path is None and geometry is None:
        return None


    return None



@log
def get_psana_corrected_data(psana_det, evt, use_default=False, dark=True,
                             common_mode=None, apply_gain_mask=True,
                             gain_mask_value=None, per_pixel_gain=False,
                             gain_mask=None, additional_gain_factor=None):
    """
    Given a psana Detector object, apply corrections as appropriate and return
    the data from the event
    @param psana_det psana Detector object
    @param evt psana event
    @param use_default If true, apply the default calibration only, using the
    psana algorithms. Otherise, use the corrections specified by the rest of
    the flags and values passed in.
    @param dark Whether to apply the detector dark, bool or numpy array
    @param common_mode Which common mode algorithm to apply. None: apply no
    algorithm. Default: use the algorithm specified in the calib folder.
    Otherwise should be a list as specified by the psana documentation for
    common mode customization
    @param apply_gain_mask Whether to apply the common mode gain mask correction
    @param gain_mask_value Multiplier to apply to the pixels, according to the
    gain mask
    @param per_pixel_gain If available, use the per pixel gain deployed to the
    calibration folder
    @param gain_mask gain mask showing which pixels to apply gain mask value
    @param additional_gain_factor Additional gain factor. Pixels counts are
    divided by this number after all other corrections.
    @return Numpy array corrected as specified.
    """

    # order is pedestals, then common mode, then gain mask, then per pixel gain

    # HACK: Force psana v2 behaviour
    PSANA2_VERSION = True


    start("psana_det.raw")
    if PSANA2_VERSION:
        # in psana2, data are stored as raw, fex, etc so the selection
        # has to be given here when the detector interface is used.
        # for now, assumes cctbx uses "raw".
        psana_det = psana_det.raw
    stop("psana_det.raw")


    if use_default:
        start("psana_det.calib")
        ret = psana_det.calib(evt)  # applies psana's complex run-dependent calibrations
        stop("psana_det.calib")
        return ret


    start("psana_det.raw_data(evt)")
    data = psana_det.raw_data(evt)
    stop("psana_det.raw_data(evt)")
    if data is None:
        return


    start("subtract psana_det.pedestals()")
    data = data.astype(np.float64)
    if isinstance(dark, bool):
        if dark:
            if PSANA2_VERSION:
                data -= psana_det.pedestals()
            else:
                data -= psana_det.pedestals(evt)
    elif isinstance( dark, np.ndarray ):
        data -= dark
    stop("subtract psana_det.pedestals()")


    if common_mode is not None and common_mode != "default":
        print("Applying common mode")

        start("psana_det.common_mode_apply(data, common_mode)")
        if common_mode == 'cspad_default':
            common_mode = (1,25,25,100,1)  # default parameters for CSPAD images
            psana_det.common_mode_apply(data, common_mode)
        elif common_mode == 'unbonded':
            common_mode = (5,0,0,0,0)  # unbonded pixels used for correction
            psana_det.common_mode_apply(data, common_mode)
        else:  # this is how it was before.. Though I think common_mode would need to be a tuple..
            psana_det.common_mode_apply(data, common_mode)
        stop("psana_det.common_mode_apply(data, common_mode)")
    else:
        print("Not applying common mode")
    

    if apply_gain_mask:
        print("Applying gain mask")

        start("apply gain mask")
        if gain_mask is None:  # TODO: consider try/except here
            gain_mask = psana_det.gain_mask(evt) == 1
        if gain_mask_value is None:
            try:
                gain_mask_value = psana_det._gain_mask_factor
            except AttributeError:
                print("No gain set for psana detector, using gain value of 1, consider disabling gain in your phil file")
                gain_mask_value = 1
        data[gain_mask] = data[gain_mask]*gain_mask_value
        stop("apply gain mask")
    else:
        print("Not applying gain mask")


    if per_pixel_gain: # TODO: test this
        start("applying psana_det.gain()")
        data *= psana_det.gain()
        stop("applying psana_det.gain()")


    if additional_gain_factor is not None:
        data /= additional_gain_factor


    return data



@log
def process_event(run, evt, psana_det):
    """
    Process a single event from a run
    @param run psana run object
    @param timestamp psana timestamp object
    """


    # HACK: Force psana v2 behaviour
    PSANA2_VERSION = True

    start("construct event timestamp")
    if PSANA2_VERSION:
        sec  = evt._seconds
        nsec = evt._nanoseconds
    else:
        time = evt.get(psana.EventId).time()
        fid = evt.get(psana.EventId).fiducials()
        sec  = time[0]
        nsec = time[1]

    ts = Event.as_timestamp(sec, nsec/1e6)
    stop("construct event timestamp")

    print("Accepted", ts)

    # HACK: these parameters have been extracted from a xtc_process run
    data = get_psana_corrected_data(psana_det, evt, use_default=False,
                                    dark=True, common_mode=None,
                                    apply_gain_mask=True, gain_mask_value=6.85,
                                    per_pixel_gain=False,
                                    additional_gain_factor=None)


    if data is None:
        print("ERROR! No data")
        return


    timestamp = t = ts
    s = t[0:4] + t[5:7] + t[8:10] + t[11:13] + t[14:16] + t[17:19] + t[20:23]
    print("Loaded shot", s)

 

@log
def test_xtc_read(ds, comm, det_name):

    for run in ds.runs():

        start(f"run.Detector({ds.det_name})")
        det = run.Detector(ds.det_name)
        stop(f"run.Detector({ds.det_name})")

        # TODO: fix flex dependency
        # if comm.Get_rank() == 0:
        #     PS_CALIB_DIR = os.environ.get('PS_CALIB_DIR')
        #     assert PS_CALIB_DIR
        #     dials_mask = easy_pickle.load(params.format.cbf.invalid_pixel_mask)
        # else:
        #     dials_mask = None
        # dials_mask = comm.bcast(dials_mask, root=0)

        start("for evt in run.events()")
        for evt in run.events():
            env_dxtbx_from_slac_metrology(run, det_name)

            process_event(run, evt, det)
        stop("for evt in run.events()")




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

    parser.add_argument("--of",
                        help="Log dir -- every rank will write its own log file")

    # Get args dict, and sanitize None types
    args         = vars(parser.parse_args())

    output_name  = args["of"]
    del args["of"]  # don't pass this to psana

    psana_kwargs = set_defaults(args, default_parameters)



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
        print("MPI Initialize, Running xtc_read Benchmark")

    start(f"psana.DataSource({psana_kwargs})")
    ds = psana.DataSource(**psana_kwargs)
    stop(f"psana.DataSource({psana_kwargs})")

    test_xtc_read(ds, comm, psana_kwargs["det_name"])


    #
    # Save log files
    #

    if rank == 0:
        print("Writing logs")

    log_path = os.path.join(output_name, f"debug_{rank}.txt")
    with open(log_path, "w") as f:
        for entry in event_log(cctbx_fmt=True):
            print(entry, file=f)
