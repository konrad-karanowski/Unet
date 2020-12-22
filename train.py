import argparse


from utils import train_model


def get_args():
    parser = argparse.ArgumentParser(description="Train UNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs',
                        dest='epochs')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=1,
                        help='Batch size',
                        dest='batch_size')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Learning rate',
                        dest='eta')
    parser.add_argument('-c',
                        '--callbacks',
                        type=int,
                        default=1,
                        help='How many epochs to save weights',
                        dest='n_callbacks')
    parser.add_argument('-m',
                        '--momentum',
                        type=float,
                        default=0.9,
                        help='Nesterov momentum value',
                        dest='momentum')
    parser.add_argument('-l',
                        '--load',
                        type=str,
                        default=None,
                        help='Path to weights (if you want to use pretrained model).',
                        dest='model_path')
    parser.add_argument('-g',
                        '--as_gray',
                        type=bool,
                        default=False,
                        help='Whether the pictures should be converted to gray (True) or coloured (False)',
                        dest='as_gray')
    parser.add_argument('-p',
                        '--patience',
                        type=int,
                        default=None,
                        help='How many epochs to tolerate no improvement.',
                        dest='patience')
    return parser.parse_args()


def main():
    args = get_args()
    train_model(
        args.epochs,
        args.batch_size,
        args.eta,
        args.momentum,
        args.n_callbacks,
        args.model_path,
        args.as_gray,
        args.patience
    )


if __name__ == '__main__':
    main()
