import os
import argparse
from progetto_cv.scripts.solver import Solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--CIFAR", type=int,default=100)
    parser.add_argument("--data_agumentation", type=int, default=0)

    # parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    # parser.add_argument("--ckpt_name", type=str, default="color")
    parser.add_argument("--print_every", type=int, default=1)

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--data_root", type=str, default="./data")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
