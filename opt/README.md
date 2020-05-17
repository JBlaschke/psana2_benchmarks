# TODOs

## Dials BCAST benchmark

The stripped down `benchmark_dials_mask_bcast` currently doesn't work due to
dependency hell:

```python
MPI Initialize, Running bcast_dials_mask Benchmark
Traceback (most recent call last):
  File "./benchmark_dials_mask_bcast.py", line 90, in <module>
    bcast_dials_mask(comm, invalid_pixel_mask_path)
  File "/Users/blaschke/Science/ExaFEL/NESAP/PSANA2/benchmarks/opt/benchmarking/perftools/event.py", line 134, in wrapper
    ret = func(*args, **kwargs)
  File "./benchmark_dials_mask_bcast.py", line 30, in bcast_dials_mask
    dials_mask = load(f)
ModuleNotFoundError: No module named 'scitbx_array_family_flex_ext'
```

Either we've got to add scitbx's flex arrays (urgh), or we just need to create
data of similar size to use as a surrogate.
