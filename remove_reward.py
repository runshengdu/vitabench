import argparse
from pathlib import Path

from vita.data_model.simulation import Results


def _read_text_with_fallback(path: Path) -> str:
    raw_text = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw_text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw_text is None:
        raw_text = path.read_text()
    return raw_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_file", type=str)
    parser.add_argument("--output", "-o", type=str, default=None)

    args = parser.parse_args()

    in_path = Path(args.simulation_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {in_path}")

    out_path = Path(args.output) if args.output is not None else in_path

    results = Results.model_validate_json(_read_text_with_fallback(in_path))

    for sim in results.simulations:
        sim.reward_info = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
