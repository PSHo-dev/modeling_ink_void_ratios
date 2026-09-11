# inkvoidmotif Studio — Quickstart

This guide gets the local Gradio Studio running and walks through one painting from
source upload to a downloadable run archive.

## 1. Install

inkvoidmotif requires Python 3.10 or newer. Open a terminal in the package directory.
In the full research repository, first run:

```bash
cd inkvoidmotif
```

Then install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Verify both entry points before configuring a provider:

```bash
inkvoidmotif --help
inkvoidmotif-studio --help
```

## 2. Configure one provider

Open `.env` and replace one placeholder with a real key:

```dotenv
OPENAI_API_KEY=your-openai-key
```

or:

```dotenv
NVIDIA_API_KEY=your-nvidia-key
```

Leave unused placeholders unchanged or remove them. Never commit or share `.env`.
Shell environment variables take precedence over values in this file.

## 3. Launch

```bash
inkvoidmotif-studio
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) if the browser does not open
automatically.

## 4. Complete the three-stage workflow

### 01 · Source

1. Upload a JPG, PNG, or WebP landscape painting.
2. Choose the number of motifs. Eight is a useful first pass.
3. Leave **Provider** on **Automatic**, or select the provider you configured.
4. Confirm the privacy and quota notice.
5. Select **Analyze painting**.

The Studio creates an editable motif plan and source-region crops. Correct names,
descriptions, roles, bounding boxes, or preservation notes before continuing.

### 02 · Motif bank

Select **Build / use motif bank** to generate a sheet from the reviewed plan. To
avoid that provider call, upload an existing motif sheet first and then select the
same button. Review the sheet and its sliced cells in the preview panel.

Editing the motif plan or replacing the uploaded sheet invalidates the active bank.
Build the bank again before composing.

### 03 · Compose

1. Describe the new landscape and give the output a short name.
2. Choose a structure and adjust **Void**, **Peak**, **Spread**, and **Seed**.
3. Inspect the live composition map in the preview panel.
4. Select **Generate new painting**.
5. Review the result, region-based void measurements, and run history.
6. Select **Download complete run** for a portable ZIP archive.

Planning, motif-sheet generation, and composition are separate provider calls.
**Cancel queued request** removes waiting work, but an active provider call may run
until its configured timeout.

## 5. Find or resume outputs

By default, each run is stored below:

```text
runs/studio/<run-id>/
```

Each retry uses a new artifact name and preserves earlier evidence. Use **Resume a
saved run** at the top of the local Studio to continue an existing run. Set
`INKVOIDMOTIF_STUDIO_RUNS` in `.env` if outputs should live elsewhere.

## Troubleshooting

**The command is not found**

Activate `.venv` and run `python -m pip install -e .` again.

**The Studio reports that no provider is ready**

Confirm that `.env` is in the directory where you launch the Studio and that a real
key replaced the placeholder. Restart the process after changing `.env`.

**Port 7860 is already in use**

```bash
inkvoidmotif-studio --port 7861
```

**Generate remains disabled**

Finish source analysis, then build or import the motif bank. If you edited the plan
or changed the sheet, rebuild the bank.

**A provider call times out**

Increase `INKVOIDMOTIF_API_TIMEOUT` in `.env` up to 900 seconds, then restart. The
failed attempt's diagnostics remain in the run folder.

**You need to share the running Studio**

Remote exposure is intentionally blocked without authentication. Configure both
`INKVOIDMOTIF_STUDIO_USER` and `INKVOIDMOTIF_STUDIO_PASSWORD`, then use `--share` or a
non-loopback `--host`. Saved-run browsing is disabled in remote mode because the
Studio is designed as a trusted single-user tool.

For CLI examples, configuration details, development checks, and release packaging,
continue with the [README](README.md).
