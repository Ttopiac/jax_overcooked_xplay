# Reports Index

Each numbered folder is one training run in chronological order. The folder name describes what makes that run distinctive.

## Runs

| Folder | What it is |
|--------|------------|
| [01_run1_mlp_3x64_baseline](01_run1_mlp_3x64_baseline/) | First 2P pass — small 3×64 MLP, quick baseline |
| [02_run2_mlp_2x256_multihri_matched](02_run2_mlp_2x256_multihri_matched/) | Scaled to 2×256 MLP with multiHRI-matched hyperparameters |
| [03_run3_perlayout_hml_eval_mep](03_run3_perlayout_hml_eval_mep/) | Per-layout H/M/L partner selection + eval-based MEP sampling |
| [04_run4_cnn_pixel_human_equivalent](04_run4_cnn_pixel_human_equivalent/) | Pixel observations (human-equivalent sprites) + IMPALA CNN, sim-to-real prep |
| [05_run3_vs_run4_crossplay_eval](05_run3_vs_run4_crossplay_eval/) | Canonical cross-play comparison of Run 3 (MLP) vs Run 4 (pixel CNN) |

Each run folder typically contains `report.md` (settings + results), `log.txt` (training output), plus any run-specific plans/logs.

## Other directories

| Folder | What it is |
|--------|------------|
| [surveys/](surveys/) | Literature surveys (ZSC/collaboration, visual RL architectures) |
| [plans/](plans/) | Pre-run design docs and settings comparisons |
| [pixel_samples/](pixel_samples/) | Rendered sprite samples across layouts and tile sizes |
| [eval_results/](eval_results/) | Raw per-pair evaluation JSONs |
| [misc/](misc/) | Cross-run analysis, experiment overview, eval tables, misc notes |
