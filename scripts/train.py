import os
import argparse
from ComputerVisionProject.scripts.solver import Solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--CIFAR", type=int,default=10)
    parser.add_argument("--data_agumentation", type=int, default=0)
    parser.add_argument("--optimizer",type=str, default="Adam")
    parser.add_argument("--plot",type=int, default=1)

    # parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    # parser.add_argument("--ckpt_name", type=str, default="color")
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--data_root", type=str, default=f"../data_{parser.parse_args().CIFAR}")

    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()


if __name__ == "__main__":
    main()
