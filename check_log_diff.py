import numpy as np
from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    # diff_helper = ReprodDiffHelper()
    # info1 = diff_helper.load_info("./diff/forward_paddle.npy")
    # info2 = diff_helper.load_info("./diff/forward_pytorch.npy")
    # path = "./diff/forward_diff.log"
    # diff_helper.compare_info(info1, info2)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-6, path=path)

    # diff_helper1 = ReprodDiffHelper()
    # info1 = diff_helper1.load_info("./diff/metric_paddle.npy")
    # info2 = diff_helper1.load_info("./diff/metric_pytorch.npy")
    # path = "./diff/metric_diff.log"
    # diff_helper1.compare_info(info1, info2)
    # diff_helper1.report(
    #     diff_method="mean", diff_threshold=1e-6, path=path)
    #

    # diff_helper2 = ReprodDiffHelper()
    # info1 = diff_helper2.load_info("./diff/loss_paddle.npy")
    # info2 = diff_helper2.load_info("./diff/loss_pytorch.npy")
    # path = "./diff/loss_diff.log"
    # diff_helper2.compare_info(info1, info2)
    # diff_helper2.report(
    #     diff_method="mean", diff_threshold=1e-6, path=path)

    # diff_helper3 = ReprodDiffHelper()
    # info1 = diff_helper3.load_info("./diff/bp_align_paddle.npy")
    # info2 = diff_helper3.load_info("./diff/bp_align_pytorch.npy")
    # path = "./diff/bp_align_diff.log"
    # diff_helper3.compare_info(info1, info2)
    # diff_helper3.report(
    #     diff_method="mean", diff_threshold=1e-6, path=path)

    diff_helper4 = ReprodDiffHelper()
    info1 = diff_helper4.load_info("./diff/train_align_paddle.npy")
    info2 = diff_helper4.load_info("./diff/train_align_benchmark.npy")
    path = "./diff/train_align_diff_log.log"
    diff_helper4.compare_info(info1, info2)
    diff_helper4.report(
        diff_method="mean", diff_threshold=1e-6, path=path)

