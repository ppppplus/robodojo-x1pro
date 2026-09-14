# Validation of the local fork candidate

- `python x1pro/check_install.py`: passed for FX001 and RM001. Both packaged robot USDs, all referenced base/configuration layers, textures, URDF chains and three RoboDojo scene assets were present. The check verifies that the LFS base layers contain binary USD data rather than pointer text.
- Python source compilation: passed for the X1 Pro examples and three integration modules.
- `git diff --cached --check`: passed for the candidate commit.
- The code-only branch excludes `x1pro/assets/`, including both approximately 240 MB `ex001_base.usd` layers and the appearance USDs. The local model files remain available in the ignored directory for validation.
- Runtime scene preview in this isolated worktree: **not completed**. Isaac Sim 5.1 remained in Kit extension startup for more than 30 minutes on this machine; the process was terminated before robot or scene creation. No rendering result should be inferred from this attempt.

The source workspace previously produced verified FX001 noodle grasp/pour episodes, but those results do not independently validate this new repository layout. Run the scene preview command in `x1pro/README.md` on a healthy Isaac Sim host before describing the fork as operational.
