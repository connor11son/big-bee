"""
Given an RGB image, generate a frogery detection mask.

@authors: Chandrakanth Gudavalli
contact: <chandrakanth@ucsb.edu>
"""

##########################################################
# Import Libraries
##########################################################

import os

from segment_wing import run_pipeline


def run_module(input_path_dict, output_folder_path, bq=None):

    ##########################################################
    # Set Paths
    ##########################################################
    # Set input arguments
    input_data = {
        "input_fname": input_path_dict["Input Image"],
        "output_fname": os.path.join(
            output_folder_path, "output.ome.tiff"
        ),  # input_path_dict["output_fname"],
    }
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if bq:
        bq.update_mex("Loading models...")

    # manipulation detection using ManTraNet
    if bq:
        bq.update_mex("Running Inference...")
    out = run_pipeline(
        input_data["input_fname"], output_masks_path=input_data["output_fname"]
    )
    outputs_path_dict = {}
    outputs_path_dict["Output Image"] = str(input_data["output_fname"])

    if bq:
        bq.update_mex("Completed")

    return outputs_path_dict


if __name__ == "__main__":
    manTraNet_root = os.path.split(os.path.dirname(__file__))[0]

    input_path_dict = {"Input Image": "/module/data/uncropped/UCSB-IZC00055302_L.JPG"}
    current_directory = os.getcwd()

    # Run algorithm and return output_paths_dict
    # print(os.path.exists(input_path_dict["Input Image"]))
    outputs_path_dict = run_module(input_path_dict, "output")

    # Get outPUT file path from dictionary
    output_img_path = outputs_path_dict["Output Image"]
    print("Image saved to:", output_img_path)
