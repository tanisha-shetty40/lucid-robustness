
import argparse

from research.robustness.brahmi_corruptions.gallery import main as create_demo_gallery


def main():
    parser = argparse.ArgumentParser(
        description="Run the image corruption library demo."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a gallery using a synthetic grayscale image.",
    )
    args = parser.parse_args()

    if args.demo:
        create_demo_gallery()
    else:
        parser.error("Please use --demo to generate the synthetic gallery.")


if __name__ == "__main__":
    main()
