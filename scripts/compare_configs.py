import sys

from llm_lab.experiments.config import diff_configs, load_config


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_configs.py BASELINE CANDIDATE")
        raise SystemExit(1)

    baseline_path = sys.argv[1]
    candidate_path = sys.argv[2]

    baseline = load_config(baseline_path)
    candidate = load_config(candidate_path)
    diffs = diff_configs(baseline, candidate)

    if not diffs:
        print("No config differences.")
        return

    for path, values in sorted(diffs.items()):
        print(f"{path}: {values['baseline']} -> {values['candidate']}")


if __name__ == "__main__":
    main()
