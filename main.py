from utils import Args, Config, parse_args, parse_env
from modules.irisRecognition import irisRecognition
import glob
from PIL import Image
import os
import numpy as np
import pandas as pd

def im_to_tmpl(fn):
    return f"templates/{os.path.splitext(os.path.basename(fn))[0]}_tmpl.npz"

def main(config: Config = Config()) -> None:
    irisRec = irisRecognition(config)

    if not config.SKIP_TEMPLATES:
        if not os.path.exists("templates"):
            os.mkdir("templates")

        # Get the list of images to process
        filename_list = []
        image_list = []
        extensions = ["bmp", "png", "gif", "jpg", "jpeg", "tiff", "tif"]
        for ext in extensions:
            for filename in glob.glob(f"data/*_im_polar/*.{ext}"):
                im = np.array(Image.open(filename))
                image_list.append(im)
                filename_list.append(filename)

        for im_polar, fn in zip(image_list, filename_list):
            vector = irisRec.extractVector(im_polar)
            np.savez_compressed(im_to_tmpl(fn), vector)

    df = pd.read_csv(config.CSV)
    df["ArcIris_score"] = np.nan

    for index, row in df.iterrows():
        enroll, search = row["Enroll"], row["Search"]
        filename1 = im_to_tmpl(enroll)
        filename2 = im_to_tmpl(search)
        vector1 = np.load(filename1)["arr_0"]
        vector2 = np.load(filename2)["arr_0"]
        score = irisRec.matchVectors(vector1, vector2)
        df.at[index, "ArcIris_score"] = round(score, 4)

    df.to_csv("scores.csv", index=False)

if __name__ == "__main__":
    args: Args = parse_args()
    config: Config = parse_env(args.env)
    main(config)