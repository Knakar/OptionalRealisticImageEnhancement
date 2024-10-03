from PIL import Image, ImageChops
import argparse

parser = argparse.ArgumentParser(description='DirectProduct Masks')
parser.add_argument("--normal", help="path to Normal mask", type=str, required=True)
parser.add_argument("--shadow", help="path to Shadow mask", type=str, required=True)
parser.add_argument("--output", help="path to Output mask", type=str, default="output.png")
args = parser.parse_args()

img1 = Image.open(args.normal).convert("1")
img2 = Image.open(args.shadow).convert("1")

assert img1.size, img2.size

direct_product = ImageChops.logical_and(img1, img2)
direct_product.save(args.output)