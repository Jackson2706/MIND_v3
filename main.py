import argparse
import os
from glob import glob

from src.data_preprocessing.behavior_processing import behavior_processing
from src.data_preprocessing.generate_data import generate_dataset
from src.data_preprocessing.news_preprocessing import news_preprocessing
from src.glove.generate_glove_dict import glovePKL
from src.mind import download_extract_small_mind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="where is the dataset path")
    parser.add_argument("--data_size", help="the size of MIND data")
    parser.add_argument("--pkl_dir", help="Where is the pkl folder")
    parser.add_argument("--glove_url", help="the url of glove representation")
    parser.add_argument("--glove_path", help="the path of glove dictionanry")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print("*" * 100)
        print("Do not find the path of dataset ---> Preparing the dataset:")
        print(f"\t Download the MIND dataset ({args.data_size})")
        train_path, validation_path = download_extract_small_mind(
            size=args.data_size, dest_path=data_dir, clean_zip_file=True
        )

        print(
            "Training path: {}\nValidation path: {}".format(
                train_path, validation_path
            )
        )
    else:
        print("Found the path of dataset ---> Skipping the dataset processing")

    # Glove dict
    print("*" * 100)
    print("Loading Glove dic")
    pkl_dir = args.pkl_dir
    glove_url = args.glove_url
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    if not os.path.exists(args.glove_path):
        glove_dict_path = glovePKL(glove_url, pkl_dir)
        print(f"Glove dict path: {glove_dict_path}")
    print("Status: Done")

    print("*" * 100)
    print("Preprocess news data")
    news_preprocessing(
        news_tsv_path=os.path.join(data_dir, "train/news.tsv"),
        glove_dir=args.glove_path,
        pkl_dir=pkl_dir,
        data_name="Train_newsID_title_embedding",
    )
    news_preprocessing(
        news_tsv_path=os.path.join(data_dir, "valid/news.tsv"),
        glove_dir=args.glove_path,
        pkl_dir=pkl_dir,
        data_name="Valid_newsID_title_embedding",
    )
    print("Status: Done")

    print("*" * 100)
    print("Preprocessing behavior data")
    train_behavior_data_path = behavior_processing(
        behavior_path=os.path.join(data_dir, "train/behaviors.tsv"),
        pkl_dir=pkl_dir,
        data_name="Train_behavior_data",
    )
    valid_behavior_data_path = behavior_processing(
        behavior_path=os.path.join(data_dir, "valid/behaviors.tsv"),
        pkl_dir=pkl_dir,
        data_name="Valid_behavior_data",
    )
    print("Status: Done")

    print("*" * 100)
    print("Generate dataset")
    generate_dataset(
        behavior_data_path=train_behavior_data_path,
        pkl_dir=pkl_dir,
        k=1,
        data_name="training_data",
    )
    generate_dataset(
        behavior_data_path=valid_behavior_data_path,
        pkl_dir=pkl_dir,
        k=1,
        data_name="validation_data",
    )
    print("Status: Done")


if __name__ == "__main__":
    main()
