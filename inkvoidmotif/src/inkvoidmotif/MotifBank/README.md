# Motif-bank data notes

`motif_bank.json` is the combined reviewed bank. `_per_painting/` preserves the
corresponding plan, generated sheet, contact sheet, and accepted motif assets
for each source.

All manifest paths are portable POSIX-style relative paths. Per-painting bank
manifests list only files present in their local `motifs/` directory. Assets
rejected during review remain under `hallucinated motifs/` for audit purposes
and are deliberately excluded from usable bank manifests.

Five source scans are not present in this snapshot, although their derived
review artifacts remain available:

- `Autumn 16 MET.jpg`
- `Autumn 17 Cleveland.jpg`
- `Spring 01 NPM.jpg`
- `Spring 06 NPM.jpg`
- `Spring 07 NPM.jpg`

Their missing source links do not affect composition from the accepted motif
assets. Re-running source analysis for those five entries requires supplying
the corresponding scans separately.
