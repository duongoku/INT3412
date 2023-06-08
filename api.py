from PIL import Image
from collections import namedtuple
from matplotlib import pyplot as plt
from os.path import join, isfile
from tqdm.auto import tqdm
import configparser
import cv2
import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import urllib.request
import warnings


warnings.filterwarnings("ignore")


from patchnetvlad.models.local_matcher import (
    calc_keypoint_centers_from_patches as calc_keypoint_centers_from_patches,
)
from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.tools.patch_matcher import PatchMatcher

IMAGES = {}
IMAGES_CACHE = {}
ITEMS = []
model, device, config = None, None, None


def apply_patch_weights(input_scores, num_patches, patch_weights):
    output_score = 0
    if len(patch_weights) != num_patches:
        raise ValueError(
            "The number of patch weights must equal the number of patches used"
        )
    for i in range(num_patches):
        output_score = output_score + (patch_weights[i] * input_scores[i])
    return output_score


def plot_two(
    cv_im_one, cv_im_two, inlier_keypoints_one, inlier_keypoints_two, plot_save_path
):
    kp_all1 = []
    kp_all2 = []
    matches_all = []
    for this_inlier_keypoints_one, this_inlier_keypoints_two in zip(
        inlier_keypoints_one, inlier_keypoints_two
    ):
        for i in range(this_inlier_keypoints_one.shape[0]):
            kp_all1.append(
                cv2.KeyPoint(
                    this_inlier_keypoints_one[i, 0].astype(float),
                    this_inlier_keypoints_one[i, 1].astype(float),
                    1,
                    -1,
                    0,
                    0,
                    -1,
                )
            )
            kp_all2.append(
                cv2.KeyPoint(
                    this_inlier_keypoints_two[i, 0].astype(float),
                    this_inlier_keypoints_two[i, 1].astype(float),
                    1,
                    -1,
                    0,
                    0,
                    -1,
                )
            )
            matches_all.append(cv2.DMatch(i, i, 0))

    im_allpatch_matches = cv2.drawMatches(
        cv_im_one,
        kp_all1,
        cv_im_two,
        kp_all2,
        matches_all,
        None,
        matchColor=(0, 255, 0),
        flags=2,
    )
    if plot_save_path is None:
        cv2.imshow("frame", im_allpatch_matches)
    else:
        im_allpatch_matches = cv2.cvtColor(im_allpatch_matches, cv2.COLOR_BGR2RGB)

        plt.imshow(im_allpatch_matches)
        # plt.show()
        plt.axis("off")
        filename = join(plot_save_path, "patchMatchings.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def match_two(
    model, device, config, im_one, im_two, plot_save_path="results", do_plot=False
):
    pool_size = int(config["global_params"]["num_pcs"])

    model.eval()

    it = input_transform(
        (
            int(config["feature_extract"]["imageresizeH"]),
            int(config["feature_extract"]["imageresizeW"]),
        )
    )

    im_one_pil = Image.fromarray(cv2.cvtColor(im_one, cv2.COLOR_BGR2RGB))
    im_two_pil = Image.fromarray(cv2.cvtColor(im_two, cv2.COLOR_BGR2RGB))

    im_one_pil = it(im_one_pil).unsqueeze(0)
    im_two_pil = it(im_two_pil).unsqueeze(0)

    input_data = torch.cat((im_one_pil.to(device), im_two_pil.to(device)), 0)

    # tqdm.write("====> Extracting Features")
    with torch.no_grad():
        image_encoding = model.encoder(input_data)

        vlad_local, _ = model.pool(image_encoding)
        # global_feats = get_pca_encoding(model, vlad_global).cpu().numpy()

        local_feats_one = []
        local_feats_two = []
        for this_iter, this_local in enumerate(vlad_local):
            this_local_feats = (
                get_pca_encoding(
                    model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))
                )
                .reshape(this_local.size(2), this_local.size(0), pool_size)
                .permute(1, 2, 0)
            )
            local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))
            local_feats_two.append(this_local_feats[1, :, :])

    # tqdm.write("====> Calculating Keypoint Positions")
    patch_sizes = [int(s) for s in config["global_params"]["patch_sizes"].split(",")]
    strides = [int(s) for s in config["global_params"]["strides"].split(",")]
    patch_weights = np.array(
        config["feature_match"]["patchWeights2Use"].split(",")
    ).astype(float)

    all_keypoints = []
    all_indices = []

    # tqdm.write("====> Matching Local Features")
    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(
            config["feature_match"], patch_size, patch_size, stride, stride
        )
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    matcher = PatchMatcher(
        config["feature_match"]["matcher"],
        patch_sizes,
        strides,
        all_keypoints,
        all_indices,
    )

    scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(
        local_feats_one, local_feats_two
    )
    score = -apply_patch_weights(scores, len(patch_sizes), patch_weights)

    if config["feature_match"]["matcher"] == "RANSAC" and do_plot:
        if plot_save_path is not None:
            tqdm.write(
                "====> Plotting Local Features and save them to "
                + str(join(plot_save_path, "patchMatchings.png"))
            )

        # using cv2 for their in-built keypoint correspondence plotting tools
        cv_im_one = cv2.resize(
            im_one,
            (
                int(config["feature_extract"]["imageresizeW"]),
                int(config["feature_extract"]["imageresizeH"]),
            ),
        )
        cv_im_two = cv2.resize(
            im_two,
            (
                int(config["feature_extract"]["imageresizeW"]),
                int(config["feature_extract"]["imageresizeH"]),
            ),
        )
        # cv2 resize slightly different from torch, but for visualisation only not a big problem

        plot_two(
            cv_im_one,
            cv_im_two,
            inlier_keypoints_one,
            inlier_keypoints_two,
            plot_save_path,
        )

    return score


def initialize_model(
    config_path="performance.ini",
):
    Option = namedtuple(
        "Option",
        ["config_path", "nocuda"],
    )
    opt = Option(
        config_path,
        False,
    )

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoder_dim, encoder = get_backend()

    # must resume to do extraction
    resume_ckpt = (
        config["global_params"]["resumePath"]
        + config["global_params"]["num_pcs"]
        + ".pth.tar"
    )

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint["state_dict"]["WPCA.0.bias"].shape[0] == int(
            config["global_params"]["num_pcs"]
        )
        config["global_params"]["num_clusters"] = str(
            checkpoint["state_dict"]["pool.centroids"].shape[0]
        )

        model = get_model(
            encoder, encoder_dim, config["global_params"], append_pca_layer=True
        )

        if int(config["global_params"]["nGPU"]) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            # if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        print(
            "=> loaded checkpoint '{}'".format(
                resume_ckpt,
            )
        )
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    return model, device, config


def initialize_all_images():
    all_images = {}
    tqdm.write("====> Loading all images")
    for i in range(len(os.listdir("images/"))):
        dir = f"images/{i}/"
        for file in os.listdir(dir):
            all_images[os.path.join(dir, file)] = get_image(os.path.join(dir, file))
            break  # Only load one image per directory
    tqdm.write("====> Done loading all images")
    return all_images


def get_image(path):
    im = cv2.imread(path, -1)
    if im is None:
        raise FileNotFoundError(path + " does not exist")
    return im


def search_for_best_match(original_image_path, model, device, config, all_images):
    original_image = get_image(original_image_path)
    results = {}
    for image_path in all_images:
        compare_image = all_images[image_path]
        tqdm.write(f"====> Comparing {original_image_path} with {image_path}")
        score = match_two(model, device, config, original_image, compare_image)
        results[image_path] = score

    max_id = max(results, key=results.get)
    return max_id, results[max_id]


def process_image(image_url):
    global IMAGES, ITEMS, model, device, config

    # Download image to temp folder
    image_path = "temp/" + image_url.split("/")[-1]
    urllib.request.urlretrieve(image_url, image_path)

    # Search for best match
    best_match_id, _ = search_for_best_match(
        image_path,
        model,
        device,
        config,
        dict(list(IMAGES.items())[: len(IMAGES) // 10]),
    )
    tqdm.write(f"====> Best match: {best_match_id}")
    best_match_id = best_match_id.split("/")[1]
    best_match_id = int(best_match_id)

    IMAGES_CACHE[image_url]["result"] = ITEMS[best_match_id]
    IMAGES_CACHE[image_url]["status"] = "done"

    tqdm.write(f"====> Done processing {image_url}")


def init():
    global IMAGES, ITEMS, model, device, config

    with open("places_and_specialties.json", encoding="utf-8") as f:
        ITEMS = json.load(f)

    IMAGES = initialize_all_images()
    model, device, config = initialize_model()

    return

    start = time.time()
    best_match_id, _ = search_for_best_match(
        f"temp/vanmieu.jpg",
        model,
        device,
        config,
        dict(list(IMAGES.items())[: len(IMAGES) // 10]),
    )
    best_match_id = best_match_id.split("/")[1]
    best_match_id = int(best_match_id)

    print(items[best_match_id])

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    init()
