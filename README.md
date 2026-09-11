# inkvoidmotif Research Artifact

This repository is an anonymized research artifact for studying void ratios in
Chinese landscape painting and generating new compositions from reusable motif
banks.

The end-user application and installable Python package are in
[`inkvoidmotif/`](inkvoidmotif/). Start with its
[`QUICKSTART.md`](inkvoidmotif/QUICKSTART.md) to install and launch the Gradio
Studio.

Repository contents:

- `Input_Images/` contains the source image collection used by the research
  workflow.
- `pixel_darkness_masks.py` and `pixel_results.csv` contain the standalone
  pixel-darkness analysis and results.
- `inkvoidmotif/` contains the tested generation pipeline, Gradio Studio,
  examples, documentation, and release instructions.

The repository intentionally contains no original version-control history,
remote URL, author identity, institutional affiliation, local credentials, or
machine-specific path metadata. See [`SUBMISSION.md`](SUBMISSION.md) for the
pre-submission verification record.

To reproduce the standalone pixel analysis, install NumPy and OpenCV in an
isolated environment and run it against one or more files or directories:

```bash
python -m pip install numpy opencv-python
python pixel_darkness_masks.py Input_Images
```

For double-anonymous review, author, affiliation, copyright-holder, and final
licensing details are withheld. See
[`inkvoidmotif/REVIEW_LICENSE.md`](inkvoidmotif/REVIEW_LICENSE.md). Source-image
rights remain with their respective institutions and rights holders.
