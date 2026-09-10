"""
Group 4 — robustness metrics requiring separate live agent run batches.

Unlike G1/G2/G3 which read from the cached outputs/runs/ directory,
every module here calls the agent directly with deliberately mutated
inputs. Enable via groups.g4_robustness: true in config.yaml.
"""