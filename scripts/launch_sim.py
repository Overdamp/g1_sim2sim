"""launch_sim: reserved CLI; runtime implementation is pending."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.exit(2, "Not implemented: see docs/implementation_plan.md and README.md.\n")


if __name__ == "__main__":
    main()
