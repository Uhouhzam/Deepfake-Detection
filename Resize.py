
import os
import argparse
from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('result_root')


def main(args):
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)
    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)
    i=0
    for file in os.listdir(args.dataset_root):
        f_img = args.dataset_root+"/"+file
        img = Image.open(f_img)
        img = img.resize((256,256))
        f_resize=args.result_root+"/"+file
        img.save(f_resize)
        i+=1
        if i % 100 == 0:
            print(i, ' resize completed.')
    
        if i == 2024-1:
            print("All", 2024, "resize completed")

if __name__ == '__main__':
    args=parser.parse_args()
    main(args)