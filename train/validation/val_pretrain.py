
import os, sys, glob, tqdm
import numpy as np
import random, time


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ""
    )
)
import torch
import MinkowskiEngine as ME
from W4DTS.core import sq_model_v3 as mdoel
from W4DTS.dataset import semanticKITTI_mutli_frame_v8_100f_valid as data
from W4DTS.utils import np_ioueval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

data_base = "/path/to/dataset"
sv_base = os.path.join(data_base,'superV_v13221')

mid_data_base = os.path.join(
    data_base,
    "mid",
)

scale = [1 / 20, 1 / 20, 1 / 20]
window_size = 1

config = data.config

data_val = data.SmanticKITTI(
    data_base,
    1,
    1,
    scale,
    sv_dir=sv_base,
    config=config,
    mid_label=mid_data_base,
    type="valid",
    sample_interval=1,
    sample_dir="sv_sample_point_v3",
    kp= 1
)

evealer = np_ioueval.iouEval(data.N_CLASSES, 0)
device = torch.device("cuda" if use_cuda else "cpu")

snap = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pretrained",
    "pretrain-v1-42-ep200.pth",
)

unet = mdoel.Model(in_channels=1, out_channels=data.N_CLASSES)
unet = unet.to(device)

print(
    "#classifer parameters",
    sum([x.nelement() for x in unet.parameters() if x.requires_grad]),
)

epoch_s = 0
evealer = np_ioueval.iouEval(data.N_CLASSES, 0)


if __name__=='__main__':
    print("Restore from " + snap)
    unet.load_state_dict(torch.load(snap))
    torch.cuda.empty_cache()

    show_time = 5
    with torch.no_grad():
        unet.eval()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        start = time.time()
        evealer.reset()

        with tqdm.tqdm(total=len(data_val) // data_val.batch_size) as pbar:
            posses = [0.0]
            min_l = 0
            for i, batch in enumerate(data_val.get_data_loader()):
                torch.cuda.empty_cache()

                locs = batch["x"][0][1].view(-1, 4)
                min_c = locs.min(0)[0].view(-1, 4)
                feats = batch["x"][1][1].view(-1, 1)
                y = batch["y_all"][1].to(device)
                x = ME.SparseTensor(
                    feats.float(), coordinates=locs, device=device
                )  # for x_in in zip(locs,feats)]
                predictions, m_f = unet(x, )
                # min_c = [min_l, min_c],
                min_l = min_c

                n_p = [
                    predictions.F[predictions.C[:, 0] == pidx][
                        batch["reverse_index"][pidx][1]
                    ]
                    for pidx in range(predictions.C[:, 0].max() + 1)
                ]
                # prediction = n_p.argmax(1)
                start_idx = 0
                for pidx in range(predictions.C[:, 0].max() + 1):
                    single_pre = n_p[pidx]
                    mid_feature = m_f.F[batch["reverse_index"][pidx][1]]

                    y_c = y[
                          start_idx: start_idx
                                     + batch["reverse_index"][pidx][1].shape[0]
                          ]
                    start_idx += batch["reverse_index"][pidx][1].shape[0]
                    evealer.addBatch(
                        single_pre.argmax(1).cpu(), y_c.int().squeeze().cpu()
                    )

                pbar.set_postfix(
                    {
                        "TnIoU": "{0:1.5f}".format(evealer.getIoU()[0])
                    }
                )  # train_loss / (i + 1) (counts)
                pbar.update(1)
        print(
            0,
            "time=",
            time.time() - start,
            "s",
        )
        m_iou, iou = evealer.getIoU()
        print("mean IOU", m_iou)
        tp, fp, fn = evealer.getStats()
        total = tp + fp + fn
        print("classes          IoU")
        print("----------------------------")
        for i in range(data.N_CLASSES):
            label_name = config["labels"][config["learning_map_inv"][i]]
            print(
                "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                    label_name, iou[i], tp[i], total[i]
                )
            )
