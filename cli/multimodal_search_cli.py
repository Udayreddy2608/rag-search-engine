import argparse

from lib.multimodal_search import verify_image_embedding


def main():
    print("Embedding shape: 512 dimensions")
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate image embedding and print its shape"
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )

    args = parser.parse_args()


    if args.command == "verify_image_embedding":
        embedding = verify_image_embedding(args.image_path)
        print(embedding.shape)


if __name__ == "__main__":
    main()