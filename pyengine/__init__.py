"""Training-side Python package. The tournament bot imports only `runtime`.

Named `pyengine` rather than `py` so it does not shadow the `py` library that
pytest imports. `python -m pytest` would fail from this repo otherwise.
"""
