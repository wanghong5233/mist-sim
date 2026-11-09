"""Export training curves from the raw result CSVs."""
import sys

from mist_sim.plotting import main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
