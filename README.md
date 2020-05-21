# PSANA2 Performance Benchmarking

Benchmarking tools to test CCTBX's xtc2 event processing pattern -- but without
relying on CCTBX's dependencies.

## Building

Recommended: build a docker image using
```bash
./build_docker.sh
```

## Benchmarks

Two benchmarks are available (these are avaiable in `/img` in the docker image)

1. `benchmark_dials_mask_bcast.py` -- benchmark the MPI bcast step for the
   dials mask. This is WIP as unpickling the data requires CCTBX's flex
   `array_family`
2. `benchmark_xtc_read.py` -- benchmark the typical way that CCTBX's xtc
   processing accesses events in from an xtc2 stream.


### Running Benchmarks

You can access information on the avialable input arguments by using the `-h`
flag. Unspecified flags default process the mini data-set in
`/img/data/xtc_test`. The only mandatory flag is the output file path, eg:
`--of=/img/output` (each mpi rank will save its own profiling data to
`<of>/debug_<rank>.txt`).


## Profiling

Profiling is done using the light-weight profiling tool in the
`opt/benchmarking` module. The idea of this is to time different parts of the
code in a light-weight (and convenient) a manner as possible. The outputed logs
will contain the start and stop times only -- postprocessing will be needed to
extract execution times. Check out `opt/benchmarking/perftools/` for more
details. 

Decorating a function with the `@log` decorator will time the whole function.
Manual timers are controlled using `start("<label>")` and `stop("<label>")`.
Timers are kept in memory, and time-stamp stings are generated on the fly when
the `event_log()` function is called. Check out `opt/test_event.py` for a
simple example.

Timing entire functions using the `@log` decorator logs the start of the
function call with the `call` status, and it logs the return time of the
function call using the `rtrn` status. Timing using the manual `start` and
`stop` timers will log the start time using the `push` status, and the stop
time using the `pop` status respectively.

Using `event_log(cctbx_fmt=True)` allows the output to be parsed using the
https://github.com/JBlaschke/cctbx_profiling tool -- some work will need to be
done to add call-stack (and push-pop semantics), but that'll be out soon.
