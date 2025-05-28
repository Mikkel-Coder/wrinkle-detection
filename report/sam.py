from pathlib import Path
import json
import matplotlib.pyplot as plt
from src.ml.test import SAMResults

plt.rcParams["font.family"] = "DejaVu Serif"


def sample_figure(sam_result: SAMResults, texture_id, image_id):
    image_path = sam_result.test_dataset_images_path / texture_id / f"{image_id}.png"
    mask_path = sam_result.test_dataset_masks_path / texture_id / f"{image_id}.png"

    sam_result._prompt_model(image_path, mask_path)

    # Probability mask
    fig, ax = plt.subplots()
    im = ax.imshow(sam_result.prob_mask, cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(
        f"report/figures/probability_{texture_id}_{image_id}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    del ax

    # Binary predicted mask
    fig, ax = plt.subplots()
    im = ax.imshow(sam_result.pred_mask, cmap="gray")
    ax.axis("off")
    fig.savefig(
        f"report/figures/mask_{texture_id}_{image_id}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    del ax


def calculate_average(data_iou):
    averages = {}
    for texture_id, image_iou_dict in data_iou.items():
        ious = list(image_iou_dict.values())
        print(ious)
        mean_iou = sum(ious) / len(ious) if ious else 0
        averages[texture_id] = mean_iou
    print(averages)
    return averages


def plot_mean_lose():

    mean_losses = [
        0.2593,
        0.2152,
        0.2021,
        0.1934,
        0.1870,
        0.1816,
        0.1769,
        0.1728,
        0.1693,
        0.1655,
        0.1626,
        0.1599,
        0.1567,
        0.1539,
        0.1520,
        0.1494,
        0.1473,
        0.1451,
        0.1429,
        0.1412,
        0.1390,
        0.1374,
        0.1357,
        0.1342,
        0.1327,
        0.1312,
        0.1298,
        0.1281,
        0.1271,
        0.1257,
        0.1241,
        0.1230,
        0.1219,
        0.1208,
        0.1196,
        0.1184,
        0.1176,
        0.1165,
        0.1153,
        0.1147,
        0.1138,
        0.1127,
        0.1116,
        0.1112,
        0.1100,
        0.1092,
        0.1083,
        0.1077,
        0.1069,
        0.1063,
        0.1056,
        0.1049,
        0.1036,
        0.1032,
        0.1029,
        0.1021,
        0.1011,
        0.1005,
        0.1002,
        0.0997,
        0.0988,
        0.0983,
        0.0976,
        0.0970,
        0.0966,
        0.0959,
        0.0954,
        0.0949,
        0.0943,
        0.0939,
        0.0936,
        0.0929,
        0.0927,
        0.0925,
        0.0920,
        0.0916,
        0.0909,
        0.0908,
        0.0905,
        0.0895,
        0.0893,
        0.0887,
        0.0886,
        0.0878,
        0.0876,
        0.0871,
        0.0868,
        0.0866,
        0.0856,
        0.0858,
        0.0854,
        0.0849,
        0.0844,
        0.0842,
        0.0840,
        0.0834,
        0.0833,
        0.0827,
        0.0825,
        0.0822,
        0.0815,
        0.0815,
        0.0811,
        0.0807,
        0.0804,
        0.0802,
        0.0796,
        0.0794,
        0.0794,
        0.0786,
        0.0787,
        0.0784,
        0.0780,
        0.0776,
        0.0775,
        0.0771,
        0.0768,
        0.0763,
        0.0763,
        0.0760,
        0.0758,
        0.0754,
        0.0752,
        0.0748,
        0.0745,
        0.0745,
        0.0742,
        0.0736,
        0.0737,
        0.0732,
        0.0733,
        0.0728,
        0.0727,
        0.0723,
        0.0722,
        0.0720,
        0.0718,
        0.0716,
        0.0713,
        0.0710,
        0.0709,
        0.0707,
        0.0704,
        0.0702,
        0.0700,
        0.0695,
        0.0693,
        0.0693,
        0.0692,
        0.0688,
        0.0690,
        0.0685,
        0.0680,
        0.0680,
        0.0680,
        0.0676,
        0.0675,
        0.0674,
        0.0671,
        0.0673,
        0.0668,
        0.0665,
        0.0664,
        0.0663,
        0.0657,
        0.0655,
        0.0656,
        0.0652,
        0.0649,
        0.0648,
        0.0648,
        0.0646,
        0.0643,
        0.0642,
        0.0642,
        0.0638,
        0.0638,
        0.0636,
        0.0633,
        0.0632,
        0.0629,
        0.0628,
        0.0627,
        0.0627,
        0.0625,
        0.0623,
        0.0620,
        0.0619,
        0.0618,
        0.0617,
        0.0613,
        0.0612,
        0.0616,
        0.0612,
        0.0606,
        0.0609,
        0.0606,
        0.0605,
        0.0600,
        0.0601,
    ]

    fig, ax = plt.subplots()
    ax.plot(mean_losses, )
    ax.set_ylabel("Mean Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.set_title("Mean Loss Learning for SAM")
    fig.savefig("report/figures/sam_learning.png")
    plt.close(fig)


def main():
    model_path = Path("src/ml/models/images_2000_epochs_200.pth")
    sam_results = SAMResults(model_path)

    data_iou_path = Path("data/iou_results.json")
    with data_iou_path.open("r") as fp:
        data_iou = json.load(fp)

    sample_figure(sam_results, texture_id="1", image_id="2")
    calculate_average(data_iou)
    plot_mean_lose()


if __name__ == "__main__":
    main()
