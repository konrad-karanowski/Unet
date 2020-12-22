import argparse


from utils import predict_mask


def get_args():
    parser = argparse.ArgumentParser(description="Train UNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',
                        '--num_classes',
                        type=int,
                        required=True,
                        help='On how many classes model was trained',
                        dest='num_classes')
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='Path to pretrained weights',
                        dest='model_path')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='Path to image or directory with images',
                        dest='input_path')
    parser.add_argument('-g',
                        '--gray',
                        type=bool,
                        default=False,
                        help='Whether input images are gray or coloured',
                        dest='as_gray'
                        )
    parser.add_argument('-t',
                        '--threshold',
                        type=float,
                        default=0.5,
                        help='Threshold for which we want to classify pixel as belonging to the class',
                        dest='threshold'
                        )
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default=None,
                        help="Where do you want to save output (if None, predictions won't be saved)",
                        dest='output')
    return parser.parse_args()


def main():
    args = get_args()
    predict_mask(
        args.as_gray,
        args.num_classes,
        args.model_path,
        args.input_path,
        args.threshold,
        args.output
    )


if __name__ == '__main__':
    main()

