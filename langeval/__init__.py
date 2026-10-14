"""Language-reliance evaluation for VLA policies on LIBERO.

Pure logic (trackers, classifiers, instruction transforms) imports only numpy. Anything touching LIBERO,
MuJoCo or a model is imported inside the function that needs it, so `pytest tests/` runs on a laptop.
"""
