# Example configs

Four reference configs are included:

| File | Purpose |
|---|---|
| `summer22_api.json` | Full API pipeline (recommended) on a summer painting, with a hand-written 10-motif plan. |
| `winter20_api.json` | Same shape on a winter painting; also shows an `ink_ratio` block enforcing a high-void (62/26/12) tonal balance on the composition. |
| `summer22_pipeline.json` | Web-workflow fallback: produces upload-ready prompts for the ChatGPT web UI instead of generating via API. |
| `winter20_pipeline.json` | Same, winter painting. |

All configs reference inputs under `examples/inputs/` so they're portable
across machines. The collaborator should drop the corresponding source
paintings here:

```text
examples/
├── README.md                    ← this file
├── summer22_api.json
├── summer22_pipeline.json
├── winter20_api.json
├── winter20_pipeline.json
└── inputs/                       ← create this folder; not included in the package
    ├── Summer 22 NPM.jpg
    ├── Winter 20 MET.jpg
    ├── motiff_output.png         (only needed for summer22_pipeline.json)
    └── quality_exemplar.png      (only needed for winter20_pipeline.json)
```

Source paintings are intentionally not shipped — museums hold copyright on
the scans and you'll have your own collection anyway. Edit the `source`
field in any config to point at your own painting and you're good to go.

To run an example end-to-end:

```bash
inkvoidmotif run examples/summer22_api.json
```

Outputs land in `runs/summer22_api/`.
