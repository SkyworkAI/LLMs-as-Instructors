import sys
import re

def extract_model_name(path):
    trimmed_path = path.rstrip('/')
    parts = trimmed_path.split('/')

    if parts[-1].isdigit():
        the_model_is = f"{parts[-2]}_{parts[-1]}"
    else:
        the_model_is = parts[-1]
    print(the_model_is)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_model_name(sys.argv[1])
    else:
        print("No path provided.")
