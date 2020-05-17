#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import psana
import numpy    as np
from   argparse import ArgumentParser
from   pickle   import dump

from benchmarking import Event, event_here, start, stop, log, event_log




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
        import socket
        raise RuntimeError(f"Could not read geometry, hostname:
                           {socket.gethostname()}")

    if metro_path is None and geometry is None:
        return None

    # These don't do any psana work => skip
    # metro = read_slac_metrology(metro_path, geometry)

    # cbf = get_cspad_cbf_handle(None, metro, 'cbf', None, "test", None, 100,
    #                            verbose = True, header_only = True)

    # from dxtbx.format.FormatCBFCspad import FormatCBFCspadInMemory
    # return FormatCBFCspadInMemory(cbf)

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


    if PSANA2_VERSION:
        # in psana2, data are stored as raw, fex, etc so the selection
        # has to be given here when the detector interface is used.
        # for now, assumes cctbx uses "raw".
        psana_det = psana_det.raw


    if use_default:
        return psana_det.calib(evt)  # applies psana's complex run-dependent calibrations


    data = psana_det.raw_data(evt)
    if data is None:
        return

    data = data.astype(np.float64)
    if isinstance(dark, bool):
        if dark:
            if PSANA2_VERSION:
                data -= psana_det.pedestals()
            else:
                data -= psana_det.pedestals(evt)
    elif isinstance( dark, np.ndarray ):
        data -= dark


    if common_mode is not None and common_mode != "default":
        if common_mode == 'cspad_default':
            common_mode = (1,25,25,100,1)  # default parameters for CSPAD images
            psana_det.common_mode_apply(data, common_mode)
        elif common_mode == 'unbonded':
            common_mode = (5,0,0,0,0)  # unbonded pixels used for correction
            psana_det.common_mode_apply(data, common_mode)
        else:  # this is how it was before.. Though I think common_mode would need to be a tuple..
            psana_det.common_mode_apply(data, common_mode)


    if apply_gain_mask:
        if gain_mask is None:  # TODO: consider try/except here
            gain_mask = psana_det.gain_mask(evt) == 1
        if gain_mask_value is None:
            try:
                gain_mask_value = psana_det._gain_mask_factor
            except AttributeError:
                print("No gain set for psana detector, using gain value of 1, consider disabling gain in your phil file")
                gain_mask_value = 1
        data[gain_mask] = data[gain_mask]*gain_mask_value


    if per_pixel_gain: # TODO: test this
        data *= psana_det.gain()


    if additional_gain_factor is not None:
        data /= additional_gain_factor


    return data



@log
def process_event(self, run, evt, psana_det):
    """
    Process a single event from a run
    @param run psana run object
    @param timestamp psana timestamp object
    """


    # HACK: Force psana v2 behaviour
    PSANA2_VERSION = True


    if PSANA2_VERSION:
        sec  = evt._seconds
        nsec = evt._nanoseconds
    else:
        time = evt.get(psana.EventId).time()
        fid = evt.get(psana.EventId).fiducials()
        sec  = time[0]
        nsec = time[1]


    # HACK: get a the time-stamp corresponding to the psana event data out of
    # the perftools event processor (instead of the cspad_tbx)
    #  ts = cspad_tbx.evt_timestamp((sec,nsec/1e6))
    evt       = Event()
    evt.t_evt = (sec, nsec/1e6)
    ts        = evt.timestamp

    
    # COMMENT: I don't think ts can be NONE
    # if ts is None:
    #     print("No timestamp, skipping shot")
    #     return


    # COMMENT: I don't think we need this either
    # if len(self.params_cache.debug.event_timestamp) > 0
    # and ts not in self.params_cache.debug.event_timestamp:
    #     return
    # self.run = run


    # COMMENT: I don't think we need this for the benchmark
    # if self.params_cache.debug.skip_processed_events or
    # self.params_cache.debug.skip_unprocessed_events or
    # self.params_cache.debug.skip_bad_events:
    #     if ts in self.known_events:
    #         if self.known_events[ts] not in ["stop", "done", "fail"]:
    #             if self.params_cache.debug.skip_bad_events:
    #                 print("Skipping event %s: possibly caused an unknown
    #                       exception previously"%ts)
    #                 return
    #         elif self.params_cache.debug.skip_processed_events:
    #             print("Skipping event %s: processed successfully previously"%ts)
    #             return
    #     else:
    #         if self.params_cache.debug.skip_unprocessed_events:
    #             print("Skipping event %s: not processed previously"%ts)
    #             return

    # self.debug_start(ts)

    # HACK: this will never get called
    # # FIXME MONA: below will be replaced with filter() callback
    # if not PSANA2_VERSION:
    #     if evt.get("skip_event") or "skip_event" in [key.key() for key in evt.keys()]:
    #         print("Skipping event",ts)
    #         self.debug_write("psana_skip", "skip")
    #         return

    print("Accepted", ts)

    # the data needs to have already been processed and put into the event by psana
    # get numpy array, 32x185x388

    # HACK: these parameters have been extracted from a xtc_process run
    data = get_psana_corrected_data(psana_det, evt, use_default=False,
                                    dark=True, common_mode=None,
                                    apply_gain_mask=None, gain_mask_value=6.85,
                                    per_pixel_gain=False,
                                    additional_gain_factor=None)


    if data is None:
        print("No data")
        self.debug_write("no_data", "skip")
        return


    # HACK: Parameters extracted from a xtc_process run show that we don't
    # enter this anyway, and I don't think that this contributes anything
    # useful to the benchmark
    # if self.params.format.cbf.override_distance is None:
    #     if self.params.format.cbf.mode == "cspad":
    #         distance = cspad_tbx.env_distance(self.params.input.address,
    #                                           run.env(),
    #                                           self.params.format.cbf.detz_offset)
    #     elif self.params.format.cbf.mode == "rayonix":
    #         distance = self.params.format.cbf.detz_offset
    #     if distance is None:
    #         print("No distance, skipping shot")
    #         self.debug_write("no_distance", "skip")
    #         return
    # else:
    #     distance = self.params.format.cbf.override_distance


    # HACK: Parameters extracted from a xtc_process run show that we don't
    # enter this. But TODO: we might want to add this anyway for benchmarking
    # if self.params.format.cbf.override_energy is None:
    #     if PSANA2_VERSION:
    #         wavelength = 12398.4187/self.psana_det.raw.photonEnergy(evt)
    #     else:
    #         wavelength = cspad_tbx.evt_wavelength(evt)
    #     if wavelength is None:
    #         print("No wavelength, skipping shot")
    #         self.debug_write("no_wavelength", "skip")
    #         return
    # else:
    #     wavelength = 12398.4187/self.params.format.cbf.override_energy


    self.timestamp = timestamp = t = ts
    s = t[0:4] + t[5:7] + t[8:10] + t[11:13] + t[14:16] + t[17:19] + t[20:23]
    print("Processing shot", s)

    # def build_dxtbx_image():
    #     if self.params.format.file_format == 'cbf':
    #         # stitch together the header, data and metadata into the final
    #         # dxtbx format object
    #         if self.params.format.cbf.mode == "cspad":
    #             dxtbx_img =
    #             cspad_cbf_tbx.format_object_from_data(self.base_dxtbx, data,
    #                                                   distance, wavelength,
    #                                                   timestamp,
    #                                                   self.params.input.address,
    #                                                   round_to_int=False)
    #         elif self.params.format.cbf.mode == "rayonix":
    #             dxtbx_img =
    #             rayonix_tbx.format_object_from_data(self.base_dxtbx, data,
    #                                                 distance, wavelength,
    #                                                 timestamp,
    #                                                 self.params.input.address)

    #         if self.params.input.reference_geometry is not None:
    #             from dxtbx.model import Detector
    #             # copy.deep_copy(self.reference_detctor) seems unsafe based on
    #             # tests. Use from_dict(to_dict()) instead.
    #             dxtbx_img._detector_instance =
    #             Detector.from_dict(self.reference_detector.to_dict())
    #             if self.params.format.cbf.mode == "cspad":
    #                 dxtbx_img.sync_detector_to_cbf() #FIXME need a rayonix version of this??

    #     elif self.params.format.file_format == 'pickle':
    #         from dxtbx.format.FormatPYunspecifiedStill import FormatPYunspecifiedStillInMemory
    #         dxtbx_img = FormatPYunspecifiedStillInMemory(image_dict)
    #     return dxtbx_img

    # dxtbx_img = build_dxtbx_image()


    # HACK: this seems to be a post-processing step => not that helpful for a
    # psana-only benchmark
    # for correction in self.params.format.per_pixel_absorption_correction:
    #     if correction.apply:
    #         if correction.algorithm == "fuller_kapton":
    #             from dials.algorithms.integration.kapton_correction import
    #             all_pixel_image_data_kapton_correction
    #             data =
    #             all_pixel_image_data_kapton_correction(image_data=dxtbx_img,
    #                                                    params=correction.fuller_kapton)()
    #             dxtbx_img = build_dxtbx_image() # repeat as necessary to update the image pixel data and rebuild the image


    # TODO: skipping for now, but since this is I/O, we maybe should include an
    # image-output step in this benchmark
    # if self.params.dispatch.dump_all:
    #     self.save_image(dxtbx_img, self.params,
    #                     os.path.join(self.params.output.output_dir, "shot-" +
    #                                  s))


    # HACK: skipping this because I have no idea what this does
    # self.cache_ranges(dxtbx_img,
    #                   self.params.input.override_spotfinding_trusted_min,
    #                   self.params.input.override_spotfinding_trusted_max)

    from dxtbx.imageset import ImageSet, ImageSetData, MemReader
    imgset = ImageSet(ImageSetData(MemReader([dxtbx_img]), None))
    imgset.set_beam(dxtbx_img.get_beam())
    imgset.set_detector(dxtbx_img.get_detector())

    if self.params.dispatch.estimate_gain_only:
        from dials.command_line.estimate_gain import estimate_gain
        estimate_gain(imgset)
        return

    # FIXME MONA: radial avg. is currently disabled
    if not PSANA2_VERSION:
        # Two values from a radial average can be stored by mod_radial_average. If present, retrieve them here
        key_low = 'cctbx.xfel.radial_average.two_theta_low'
        key_high = 'cctbx.xfel.radial_average.two_theta_high'
        tt_low = evt.get(key_low)
        tt_high = evt.get(key_high)

    if self.params.radial_average.enable:
        if tt_low is not None or tt_high is not None:
            print("Warning, mod_radial_average is being used while also using xtc_process radial averaging. mod_radial_averaging results will not be logged to the database.")

    from dxtbx.model.experiment_list import ExperimentListFactory
    experiments = ExperimentListFactory.from_imageset_and_crystal(imgset, None)

    try:
        self.pre_process(experiments)
    except Exception as e:
        self.debug_write("preprocess_exception", "fail")
        return

    if not self.params.dispatch.find_spots:
        self.debug_write("data_loaded", "done")
        return

    # before calling DIALS for processing, set output paths according to the templates
    if not self.params.output.composite_output:
        if self.indexed_filename_template is not None and "%s" in self.indexed_filename_template:
            self.params.output.indexed_filename = os.path.join(self.params.output.output_dir, self.indexed_filename_template%("idx-" + s))
        if "%s" in self.refined_experiments_filename_template:
            self.params.output.refined_experiments_filename = os.path.join(self.params.output.output_dir, self.refined_experiments_filename_template%("idx-" + s))
        if "%s" in self.integrated_filename_template:
            self.params.output.integrated_filename = os.path.join(self.params.output.output_dir, self.integrated_filename_template%("idx-" + s))
        if "%s" in self.integrated_experiments_filename_template:
            self.params.output.integrated_experiments_filename = os.path.join(self.params.output.output_dir, self.integrated_experiments_filename_template%("idx-" + s))
        if "%s" in self.coset_filename_template:
            self.params.output.coset_filename = os.path.join(self.params.output.output_dir, self.coset_filename_template%("idx-" + s,
                self.params.integration.coset.transformation))
        if "%s" in self.coset_experiments_filename_template:
            self.params.output.coset_experiments_filename = os.path.join(self.params.output.output_dir, self.coset_experiments_filename_template%("idx-" + s,
                self.params.integration.coset.transformation))
        if "%s" in self.reindexedstrong_filename_template:
            self.params.output.reindexedstrong_filename = os.path.join(self.params.output.output_dir, self.reindexedstrong_filename_template%("idx-" + s))

    if self.params.input.known_orientations_folder is not None:
        expected_orientation_path = os.path.join(self.params.input.known_orientations_folder, os.path.basename(self.params.output.refined_experiments_filename))
        if os.path.exists(expected_orientation_path):
            print("Known orientation found")
            from dxtbx.model.experiment_list import ExperimentListFactory
            self.known_crystal_models = ExperimentListFactory.from_json_file(expected_orientation_path, check_format=False).crystals()
        else:
            print("Image not previously indexed, skipping.")
            self.debug_write("not_previously_indexed", "stop")
            return

    # Load a dials mask from the trusted range and psana mask
    from dials.util.masking import MaskGenerator
    generator = MaskGenerator(self.params.border_mask)
    mask = generator.generate(imgset)
    if self.params.format.file_format == "cbf" and self.dials_mask is not None:
        mask = tuple([a&b for a, b in zip(mask,self.dials_mask)])
    if self.spotfinder_mask is None:
        self.params.spotfinder.lookup.mask = mask
    else:
        self.params.spotfinder.lookup.mask = tuple([a&b for a, b in zip(mask,self.spotfinder_mask)])

    self.debug_write("spotfind_start")
    try:
        observed = self.find_spots(experiments)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(str(e), "event", timestamp)
        self.debug_write("spotfinding_exception", "fail")
        return

    print("Found %d bright spots"%len(observed))

    if self.params.dispatch.hit_finder.enable and len(observed) < self.params.dispatch.hit_finder.minimum_number_of_reflections:
        print("Not enough spots to index")
        self.debug_write("not_enough_spots_%d"%len(observed), "stop")
        return
    if self.params.dispatch.hit_finder.maximum_number_of_reflections is not None:
        if self.params.dispatch.hit_finder.enable and len(observed) > self.params.dispatch.hit_finder.maximum_number_of_reflections:
            print("Too many spots to index - Possibly junk")
            self.debug_write("too_many_spots_%d"%len(observed), "stop")
            return

    self.restore_ranges(dxtbx_img)

    # save cbf file
    if self.params.dispatch.dump_strong:
        self.save_image(dxtbx_img, self.params, os.path.join(self.params.output.output_dir, "hit-" + s))

        # save strong reflections.  self.find_spots() would have done this, but we only
        # want to save data if it is enough to try and index it
        if self.strong_filename_template:
            if "%s" in self.strong_filename_template:
                strong_filename = self.strong_filename_template%("hit-" + s)
            else:
                strong_filename = self.strong_filename_template
            strong_filename = os.path.join(self.params.output.output_dir, strong_filename)

            from dials.util.command_line import Command
            Command.start('Saving {0} reflections to {1}'.format(
                    len(observed), os.path.basename(strong_filename)))
            observed.as_pickle(strong_filename)
            Command.end('Saved {0} observed to {1}'.format(
                    len(observed), os.path.basename(strong_filename)))

    if not self.params.dispatch.index:
        self.debug_write("strong_shot_%d"%len(observed), "done")
        return

    # index and refine
    self.debug_write("index_start")
    try:
        experiments, indexed = self.index(experiments, observed)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(str(e), "event", timestamp)
        self.debug_write("indexing_failed_%d"%len(observed), "stop")
        return

    if self.params.dispatch.dump_indexed:
        img_path = self.save_image(dxtbx_img, self.params, os.path.join(self.params.output.output_dir, "idx-" + s))
        imgset = ExperimentListFactory.from_filenames([img_path]).imagesets()[0]
        assert len(experiments.detectors()) == 1;       imgset.set_detector(experiments[0].detector)
        assert len(experiments.beams()) == 1;               imgset.set_beam(experiments[0].beam)
        assert len(experiments.scans()) <= 1;               imgset.set_scan(experiments[0].scan)
        assert len(experiments.goniometers()) <= 1; imgset.set_goniometer(experiments[0].goniometer)
        for expt_id, expt in enumerate(experiments):
            expt.imageset = imgset

    self.debug_write("refine_start")

    try:
        experiments, indexed = self.refine(experiments, indexed)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(str(e), "event", timestamp)
        self.debug_write("refine_failed_%d"%len(indexed), "fail")
        return

    if self.params.dispatch.reindex_strong:
        self.debug_write("reindex_start")
        try:
            self.reindex_strong(experiments, observed)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(str(e), "event", timestamp)
            self.debug_write("reindexstrong_failed_%d"%len(indexed), "fail")
            return

    if not self.params.dispatch.integrate:
        self.debug_write("index_ok_%d"%len(indexed), "done")
        return

    # integrate
    self.debug_write("integrate_start")
    self.cache_ranges(dxtbx_img, self.params.input.override_integration_trusted_min, self.params.input.override_integration_trusted_max)

    if self.cached_ranges is not None:
        # Load a dials mask from the trusted range and psana mask
        imgset = ImageSet(ImageSetData(MemReader([dxtbx_img]), None))
        imgset.set_beam(dxtbx_img.get_beam())
        imgset.set_detector(dxtbx_img.get_detector())
        from dials.util.masking import MaskGenerator
        generator = MaskGenerator(self.params.border_mask)
        mask = generator.generate(imgset)
        if self.params.format.file_format == "cbf" and self.dials_mask is not None:
            mask = tuple([a&b for a, b in zip(mask,self.dials_mask)])
    if self.integration_mask is None:
        self.params.integration.lookup.mask = mask
    else:
        self.params.integration.lookup.mask = tuple([a&b for a, b in zip(mask,self.integration_mask)])

    try:
        integrated = self.integrate(experiments, indexed)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(str(e), "event", timestamp)
        self.debug_write("integrate_failed_%d"%len(indexed), "fail")
        return
    self.restore_ranges(dxtbx_img)

 




@log
def test_xtc_read(ds, comm):

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

        for evt in run.events():
            env_dxtbx_from_slac_metrology(run, params.input.address)

            process_event(run, evt)

    ims.finalize()




def get_calib_data(ds):

    calib_data = list()
    for run in ds.runs():
        print(f"Getting Calibration for run: {run.runnum}")
        calib_data.append(run.calibconst)

    return calib_data



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
