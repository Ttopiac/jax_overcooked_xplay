Wrap up experiment results for the current run. Do the following:

1. Check `reports/eval_results/` for JSON files. If empty, say "No experiment results found." and stop.

2. Check `checkpoints/` for checkpoint files to determine which methods were trained.

3. Run `python print_tables.py` to display current results in terminal.

4. Determine the run number by checking existing `reports/run*_report.md` files and incrementing.

5. Create `reports/run{N}_report.md` with:
   - Full settings table (read from checkpoint configs and training scripts)
   - Table 1: Population self-play results
   - Table 2: Ego self-play results per layout
   - Table 3: Cross-play summary
   - Key observations
   - Training timeline (if `reports/run{N}_log.txt` exists, extract step durations)
   - How to resume instructions
   - Known issues (if any)
   - Files section listing checkpoint and eval result locations

6. Update `reports/settings_comparison.md` to include the new run alongside previous runs.

7. Update `reports/EXPERIMENTS.md` to add the new run entry with date, file locations, key settings, and best cross-play result.

8. Git commit the new/updated report files.

9. Show a summary of the results to the user.

If argument is provided (e.g. `/wrap-exp "Run 3 - per-env partner sampling"`), use it as the run description. Otherwise infer from changes.
