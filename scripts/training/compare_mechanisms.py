"""Run HPGI and joint PPO from one common initialization."""
import sys

from mist_sim.comparison import main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
