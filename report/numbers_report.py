import numpy as np


def main():
    sam = np.array([17.13, 28.31, 20.01, 18.58, 25.24, 9.62, 26.57, 6.07, 28.04, 10.82])
    gabor = np.array([4.00, 17.32, 4.93, 18.10, 13.91, 2.16, 7.19, 0.88, 3.67, 1.89])
    improvement = np.array([328, 63, 306, 3, 81, 345, 270, 590, 664, 472])

    mean_sam = np.mean(sam)
    std_sam = np.std(sam)
    var_sam = np.var(sam)

    mean_gabor = np.mean(gabor)
    std_gabor = np.std(gabor)
    var_gabor = np.var(gabor)

    mean_improv = np.mean(improvement)
    std_improv = np.std(improvement)
    var_improv = np.var(improvement)

    print(f"SAM: mean={mean_sam:.2f}, std={std_sam:.2f}, var={var_sam:.2f}")
    print(f"Gabor: mean={mean_gabor:.2f}, std={std_gabor:.2f}, var={var_gabor:.2f}")
    print(
        f"Improvement: mean={mean_improv:.2f}, std={std_improv:.2f}, var={var_improv:.2f}"
    )


if __name__ == "__main__":
    main()
