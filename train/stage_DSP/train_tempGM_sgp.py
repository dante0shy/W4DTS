
import os, sys, glob, tqdm
import numpy as np
import random, time


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ""
    )
)
import torch
import json, shutil, yaml
import MinkowskiEngine as ME
from MinkowskiEngine.utils import sparse_quantize
import torch.nn as nn
import torch.optim as optim
from W4DTS.core import sq_model_v3 as mdoel
from W4DTS.utils.pesudo_label_generator import (
    PesudoLabelGenerator_v9_8_rev_s as PesudoLabelGenerator,
)

from W4DTS.dataset import semanticKITTI_mutli_frame_v8_100f_train as data
from W4DTS.dataset import semanticKITTI_mutli_frame_v8_100f_rev as data_lseq
from W4DTS.utils import np_ioueval

torch.multiprocessing.set_sharing_strategy("file_descriptor")

config_yaml = yaml.load(
    open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "{}.yaml".format(os.path.abspath(__file__).split("/")[-1].split(".")[0]),
        )
    ),
    Loader=yaml.CLoader
)

os.environ["CUDA_VISIBLE_DEVICES"] = config_yaml["CUDA_VISIBLE_DEVICES"]
use_cuda = torch.cuda.is_available()


data_base = config_yaml["data_base"]
sv_dir = config_yaml["sv_dir"]
log_pos = config_yaml["log_pos"]
os.makedirs(log_pos,exist_ok=True)

scale = [1 / 20, 1 / 20, 1 / 20]
batch_size = config_yaml["batch_size"]
sample_interval = config_yaml["sample_interval"]
window_size = config_yaml["sample_interval"]
update_rate = config_yaml["update_rate"]
config = data.config
kp = 7
pretrain = True
training_epochs = config_yaml["training_epochs"]

mid_data_base = os.path.join(
    data_base,
    config_yaml["mid_data_base_format"].format(
        os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    ),
)



data_update = data_lseq.SmanticKITTI(
    data_base,
    4,
    window_size,
    scale,
    sv_dir=sv_dir,
    mid_label=mid_data_base,
    config=config,
    sample_interval=sample_interval,
    sample_dir="sv_sample_point_v3",
    kp = kp +1
)
data_train = data.SmanticKITTI(
    data_base,
    batch_size,
    1,
    scale,
    sv_dir=sv_dir,
    mid_label=mid_data_base,
    config=config,
    sample_interval=sample_interval,
    sample_dir="sv_sample_point_v3",
    kp=1

)
data_val = data.SmanticKITTI(
    data_base,
    batch_size,
    1,
    scale,
    sv_dir=sv_dir,
    config=config,
    mid_label=mid_data_base,
    type="valid",
    sample_interval=1,
    sample_dir="sv_sample_point_v3",
    kp= 1
)

evealer = np_ioueval.iouEval(data.N_CLASSES, 0)
evealer_up = np_ioueval.iouEval(data.N_CLASSES, 0)

evealer_svrefine = np_ioueval.iouEval(data.N_CLASSES, 0)
device = torch.device("cuda" if use_cuda else "cpu")


log_pos = os.path.join(
    log_pos, os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
)
if not os.path.exists(log_pos):
    os.makedirs(log_pos, exist_ok=True)
log_path = os.path.join(log_pos, "snap")
if not os.path.exists(log_path):
    os.mkdir(log_path)
exp_name = "unet_{}_GM".format(data_train.scale[0])
log_dir = os.path.join(log_path, exp_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logs_dir = os.path.join(log_dir, "log.json")
if os.path.exists(logs_dir):
    logs = json.load(open(logs_dir, "r"))
else:
    logs = {}


unet = mdoel.Model(in_channels=1, out_channels=data.N_CLASSES)
pl_generator = PesudoLabelGenerator(
    in_channels=unet.PLANES[-1], out_channels=data.N_CLASSES
)
unet = unet.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()))

print(
    "#classifer parameters",
    sum([x.nelement() for x in unet.parameters() if x.requires_grad]),
)

pretrain_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pretrained",
    "model-b-spp-tempOT.pth",
)
epoch_s = 0
snap = glob.glob(os.path.join(log_dir, "net*.pth"))
snap = list(sorted(snap, key=lambda x: int(x.split("-")[-1].split(".")[0])))
train_first = True
if pretrain and not snap:
    print("Pertrained from {}".format(pretrain_dir))
    model_dict = unet.state_dict()
    pretrained_dict = {
        k: v for k, v in torch.load(pretrain_dir).items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    unet.load_state_dict(model_dict)
elif snap:
    print("Restore from " + snap[-1])
    unet.load_state_dict(torch.load(snap[-1]))
    epoch_s = int(snap[-1].split("/")[-1].split(".")[0].split("-")[-1])
    optimizer.load_state_dict(torch.load(snap[-1].replace("net-", "optim-")))
    train_first = False

for i in range(20):
    os.makedirs(os.path.join(mid_data_base, "{:02d}".format(i)), exist_ok=True)

sys.stdout.flush()
skip_val = True
for epoch in range(epoch_s, training_epochs):

    stats = {}
    start = time.time()
    train_loss = 0
    batch_loss = []
    mid = time.time()
    # for i,batch in enumerate(data.train_data_loader):
    if str(epoch) not in logs.keys():
        logs[str(epoch)] = {"epoch": epoch, "TrainLoss": 0, "mIoU": 0, "ValData": ""}
    show_time = 5

    if not epoch % update_rate and train_first:
        torch.multiprocessing.set_sharing_strategy('file_system')
        evealer_up.reset()
        unet.eval()
        pl_generator.eval()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if os.path.exists(mid_data_base):
            shutil.rmtree(mid_data_base)

        for i in range(20):
            os.makedirs(os.path.join(mid_data_base, "{:02d}".format(i)), exist_ok=True)
        start = time.time()
        counts_t = 0
        counts_a = 0
        with torch.no_grad():
            with tqdm.tqdm(
                total=len(data_update.files)*(sample_interval//2+1)// data_update.batch_size#// data_train.batch_size
            ) as pbar:
                time_s = time.time()

                for i, batch_loader in enumerate(data_update.get_data_loader()):
                    sur_sv_feature_list = [None for tmp in range(data_update.batch_size)]
                    sur_sv_feature_coords_list = [
                        None for tmp in data_update.current_data
                    ]
                    sur_sv_gt_list = [None for tmp in data_update.current_data]
                    lgs = [None for tmp in data_update.current_data]
                    sur_sv_lg_feat_list = [None for tmp in data_update.current_data]
                    ini_mean_features = [None for tmp in data_update.current_data]
                    ini_ori_coords = [None for tmp in data_update.current_data]
                    ini_mid_gt_sv = [None for tmp in data_update.current_data]
                    sm_mean_features = [None for tmp in data_update.current_data]
                    sm_ori_coords = [None for tmp in data_update.current_data]
                    sm_mid_gt_sv = [None for tmp in data_update.current_data]
                    sf_mean_features = [None for tmp in data_update.current_data]
                    sf_ori_coords = [None for tmp in data_update.current_data]
                    poses = []
                    for idx, batch in enumerate(batch_loader):
                        torch.cuda.empty_cache()
                        locs = batch["x"][0][0]
                        feats = batch["x"][1][0]
                        x = ME.SparseTensor(
                            feats.float(), coordinates=locs, device=device
                        )
                        y = batch["y"][0].to(device)
                        if not idx:
                            poses = [
                                torch.cat(batch["poses"], dim=0).float(),
                                torch.cat(batch["poses"], dim=0).float(),
                            ]
                        else:
                            poses.pop(0)
                            poses.append(torch.cat(batch["poses"], dim=0).float())
                            # poses[] = []

                        predictions, mid_features = unet(x,)
                        n_p = [
                            predictions.F[predictions.C[:, 0] == pidx][
                                batch["reverse_index"][pidx][0]
                            ]
                            for pidx in range(predictions.C[:, 0].max() + 1)
                        ]

                        mid_f = [
                            mid_features.F[predictions.C[:, 0] == pidx][
                                batch["reverse_index"][pidx][0]
                            ]
                            for pidx in range(mid_features.C[:, 0].max() + 1)
                        ]

                        ##TODO: update machinism
                        start_idx = 0
                        sk = 0
                        for pidx in range(predictions.C[:, 0].max() + 1):
                            torch.cuda.empty_cache()

                            fid = batch["idx"][pidx]
                            seq_id = batch["seq_idx"][pidx]
                            # infile = data_update.files[seq_id][1][pos]
                            infile = data_update.current_data[seq_id][1][fid]
                            seq_idx = infile.split("/")[-3]
                            f_idx = infile.split("/")[-1].split(".")[0]
                            single_pre = n_p[pidx]
                            mid_feature = mid_f[pidx]
                            mid_gt = y[
                                start_idx : start_idx
                                + batch["reverse_index"][pidx][0].shape[0]
                            ]
                            mid_coord = batch
                            sv_info = batch["svs"][pidx]
                            mid_sv = batch["mid_sv"][pidx]

                            sv_info[0] = [sv_info[0].flatten()]#.int().numpy()
                            sv_info[1] = [sv_info[1].flatten()]#.int().numpy()
                            sv_info[2] = [sv_info[2].flatten()]#.int().numpy()
                            sv_info[3] = torch.unique(torch.cat(sv_info[3]).long()).long()

                            sv_prob = pl_generator.sv_group(single_pre, sv_info)
                            mean_features = pl_generator.sv_group(mid_feature, sv_info)
                            ori_coords = pl_generator.sv_group(
                                batch["o_coords"][pidx][0].cuda(), sv_info
                            )
                            mid_gt_sv = pl_generator.sv_group_label(
                                mid_gt, sv_info
                            ).max(1)[1]
                            gt_all = batch["y_all"][pidx][0].view(-1)

                            sv_gt = (
                                        torch.zeros([sv_info[0][0].shape[0], data.N_CLASSES])
                                            .cuda()
                                            .index_add_(
                                            0,
                                            torch.LongTensor(sv_info[1][0]).cuda().long(),
                                            torch.eye(data.N_CLASSES)[
                                                gt_all.long()
                                            ].cuda(),
                                        )
                                    ) / torch.Tensor(sv_info[2][0].float()).cuda().view(-1, 1)


                            with torch.no_grad():
                                torch.cuda.empty_cache()

                                if idx:

                                    (
                                        trust_svm,
                                        sur_sv_feature,
                                        sur_sv_feature_coords,
                                        sur_sv_gt,
                                        uw,
                                        sv_prob_updated,
                                        sm_index
                                    ) = pl_generator(
                                        sur_sv_feature_list[pidx].cuda(),
                                        sur_sv_feature_coords_list[pidx].cuda(),
                                        sur_sv_gt_list[pidx].cuda(),
                                        # sv_info[3],
                                        sv_prob,
                                        mean_features,
                                        ori_coords,
                                        [poses[0][pidx], poses[1][pidx]],
                                        sv_gt,
                                        lgs[pidx],
                                        val = True,
                                        tmp_flag = True,#sv_gt#True if epoch else False
                                        idx = idx,
                                        sf= [sf_mean_features[pidx],sf_ori_coords[pidx],],
                                        rev=batch["re"][pidx]

                                    )
                                else:
                                    lgs[pidx] = (
                                        mid_gt_sv != 0
                                    ).sum()  # len(sv_info[3])

                                    (
                                        trust_svm,
                                        sur_sv_feature,
                                        sur_sv_feature_coords,
                                        sur_sv_gt,
                                        uw,
                                        sv_prob_updated,
                                        sm_index
                                    ) = pl_generator(
                                        mean_features[mid_gt_sv != 0],
                                        ori_coords[mid_gt_sv != 0],
                                        torch.eye(pl_generator.out_channels)[mid_gt_sv[mid_gt_sv != 0].long().view(-1)].cuda(),
                                        # sv_info[3],
                                        sv_prob,
                                        mean_features,
                                        ori_coords,
                                        [poses[0][pidx], poses[1][pidx]],
                                        sv_gt,
                                        lgs[pidx],
                                        val=True,
                                        tmp_flag = False,
                                        rev=batch["re"][pidx]
                                    )
                                    ini_mean_features[pidx] = mean_features[mid_gt_sv != 0].detach_().cpu()
                                    ini_ori_coords[pidx] = ori_coords[mid_gt_sv != 0].detach_().cpu()
                                    ini_mid_gt_sv[pidx] = mid_gt_sv[mid_gt_sv != 0].detach_().cpu()
                                    pass
                            label_sv = sv_prob.max(1)



                            o_start_idx = start_idx
                            start_idx += batch["reverse_index"][pidx][0].shape[0]
                            sm_mean_features[pidx] = mean_features[sm_index[:lgs[pidx]]].detach_().cpu()
                            sm_ori_coords[pidx] = ori_coords[sm_index[:lgs[pidx]]].detach_().cpu()
                            sm_mid_gt_sv[pidx] = (sv_prob_updated )[sm_index[:lgs[pidx]]].detach_().cpu()#.max(1)[1]
                            if sur_sv_feature.shape[0] and sur_sv_feature.shape[0] > 10000:
                                sample_idx = torch.randperm(sur_sv_feature.shape[0])[
                                    :10000
                                ].long()

                                (
                                    sur_sv_feature_list[pidx],
                                    sur_sv_feature_coords_list[pidx],
                                    sur_sv_gt_list[pidx],
                                ) = (
                                    torch.clone(
                                        sur_sv_feature[sample_idx].detach_().cpu()
                                    ),
                                    torch.clone(
                                        sur_sv_feature_coords[sample_idx]
                                        .detach_()
                                        .cpu()
                                    ),
                                    torch.clone(sur_sv_gt[sample_idx].detach_().cpu()),
                                )
                            elif sur_sv_feature.shape[0]:
                                (
                                    sur_sv_feature_list[pidx],
                                    sur_sv_feature_coords_list[pidx],
                                    sur_sv_gt_list[pidx],
                                ) = (
                                    torch.clone(sur_sv_feature.detach_().cpu()),
                                    torch.clone(sur_sv_feature_coords.detach_().cpu()),
                                    torch.clone(sur_sv_gt.detach_().cpu()),
                                )
                            if sur_sv_feature_list[pidx].shape[0]< 10000:
                                sample_idx = torch.randperm(mean_features.shape[0])[
                                             :10000 - sur_sv_feature_list[pidx].shape[0]
                                             ].long()
                                tmp1 = torch.zeros([mean_features.shape[0]], dtype=torch.bool)
                                tmp1[sample_idx] = True
                                tmp1[sm_index[:lgs[pidx]]] = False
                                tmp1[trust_svm] = False
                                (
                                    sf_mean_features[pidx],
                                    sf_ori_coords[pidx],
                                ) = (
                                    torch.clone(
                                        mean_features[sample_idx].detach_().cpu()
                                    ),
                                    torch.clone(
                                        ori_coords[sample_idx]
                                            .detach_()
                                            .cpu()
                                    ),
                                )
                            sur_sv_feature_list[pidx] = torch.cat([
                                sm_mean_features[pidx],
                                # ini_mean_features[pidx],
                                sur_sv_feature_list[pidx]
                            ], dim=0)
                            sur_sv_feature_coords_list[pidx] = torch.cat([
                                sm_ori_coords[pidx],
                                # ini_ori_coords[pidx],
                                sur_sv_feature_coords_list[pidx]
                            ], dim=0)
                            sur_sv_gt_list[pidx] = torch.cat([
                                sm_mid_gt_sv[pidx],
                                # ini_mid_gt_sv[pidx],
                                sur_sv_gt_list[pidx]
                            ], dim=0)

                            if not len(trust_svm):
                                sk+=1

                                continue
                            label_trust_sv = (sv_prob_updated )[trust_svm.flatten()].max(1)[1]
                            # label_trust_sv = sv_prob[trust_svm.flatten()].max(1)[1]
                            sv_gt_trust = sv_gt.max(dim=1)[1][trust_svm.flatten()]
                            gt_all = batch["y_all"][pidx][0].view(-1)
                            sv_gt = (
                                torch.zeros([sv_info[0][0].shape[0], data.N_CLASSES])
                                .cuda()
                                .index_add_(
                                    0,
                                    torch.LongTensor(sv_info[1][0]).cuda().long(),
                                    torch.eye(data.N_CLASSES)[
                                        gt_all.long()
                                    ].cuda(),
                                )
                            ) / torch.Tensor(sv_info[2][0].float()).cuda().view(-1, 1)
                            # label_trust_sv = label_sv[1][trust_svm]

                            counts_t += sv_gt_trust.shape[0]#.item()
                            counts_a += (sv_gt_trust == label_trust_sv).sum().item()
                            evealer_up.addBatch(label_trust_sv.cpu(),sv_gt_trust.cpu())

                            out_mid = np.hstack(
                                [
                                    trust_svm.view(-1, 1).cpu().numpy(),
                                    label_trust_sv.view(-1, 1).cpu().numpy(),
                                ]
                            )
                            out_mid_idx = np.logical_not(
                                np.isin(out_mid[:, 0], sv_info[3])
                            )
                            if not out_mid_idx.sum():
                                continue
                            # out_mid = out_mid[out_mid_idx]
                            out_dir = os.path.join(
                                mid_data_base,
                                data_train.mid_data_format.format(seq_idx, f_idx),
                            )
                            out_mid = out_mid[out_mid_idx]

                            if os.path.exists(out_dir):

                                with open(
                                        out_dir,
                                        "rb",
                                ) as f:

                                    ores = np.load(f)

                                t = np.logical_not(np.isin(out_mid[:, 0], ores[:, 0]))
                                if np.sum(t):
                                    out_mid = np.vstack([ores, out_mid[t]])
                                else:
                                    out_mid = ores
                            if os.path.exists(out_dir):
                                os.remove(out_dir)
                            with open(
                                out_dir,
                                "wb",
                            ) as f:

                                np.save(
                                    f, out_mid,
                                )
                        x = evealer_up.getIoU()[1][1:9]
                        pbar.set_postfix(
                            {"Tacc": "{0:1.3f}".format(counts_a / counts_t)}
                        )  # train_loss / (i + 1) (counts)
                        pbar.update(1)
                        del batch
                        if sk >=2:
                            break
        logs[str(epoch)]["update_acc"] = float(counts_a / counts_t)
        json.dump(logs, open(logs_dir, "w"))
        torch.multiprocessing.set_sharing_strategy('file_descriptor')

    if train_first:
        unet.train()
        pl_generator.eval()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        train_loss = 0
        counts = 0

        with tqdm.tqdm(total=len(data_train) // data_train.batch_size) as pbar:
            time_s = time.time()
            for i, batch in enumerate(data_train.get_data_loader()):
                for idx, pos in enumerate(range(1, data_train.window_size + 1)):
                    torch.cuda.empty_cache()

                    optimizer.zero_grad()
                    start = pos - 1
                    end = pos
                    locs = batch["x"][0][end]
                    feats = batch["x"][1][end]
                    y = batch["y"][end].to(device)
                    poses = batch["poses"][start : end + 1]
                    x = ME.SparseTensor(feats.float(), coordinates=locs, device=device)

                    predictions, mid_features = unet(x,)
                    n_p = [
                        predictions.F[predictions.C[:, 0] == pidx][
                            batch["reverse_index"][pidx][pos]
                        ]
                        for pidx in range(predictions.C[:, 0].max() + 1)
                    ]
                    loss = criterion(torch.cat(n_p, dim=0), y.long().flatten())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()  #
                    counts += 1
                pbar.set_postfix(
                    {"TLoss": "{0:1.5f}".format(train_loss / (counts)),}
                )  # train_loss / (i + 1) (counts)
                pbar.update(1)
                del batch
                del loss
                del n_p

        print(
            epoch,
            "{:02d} Train loss".format(epoch),
            train_loss / counts,
            "time=",
            time.time() - start,
            "s",
        )
        torch.save(
            unet.state_dict(), os.path.join(log_dir, "net-%09d" % epoch + ".pth")
        )
        torch.save(
            optimizer.state_dict(), os.path.join(log_dir, "optim-%09d" % epoch + ".pth")
        )
        logs[str(epoch)]["TrainLoss"] = float(train_loss / counts)
        json.dump(logs, open(logs_dir, "w"))
    else:
        print("test first")
        train_first = True
    sys.stdout.flush()

    if not skip_val:  # not train_first:
        with torch.no_grad():
            unet.eval()
            pl_generator.eval()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            start = time.time()
            evealer.reset()
            evealer_svrefine.reset()

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
                    predictions, m_f = unet(x,)
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
                        sv_info = batch["svs"][pidx][1]

                        y_c = y[
                            start_idx : start_idx
                            + batch["reverse_index"][pidx][1].shape[0]
                        ]
                        start_idx += batch["reverse_index"][pidx][1].shape[0]
                        evealer.addBatch(
                            single_pre.argmax(1).cpu(), y_c.int().squeeze().cpu()
                        )

                    pbar.set_postfix(
                        {
                            "TnIoU": "{0:1.5f}".format(evealer.getIoU()[0]),
                            "uTnIoU": "{0:1.5f}".format(evealer_svrefine.getIoU()[0]),
                        }
                    )  # train_loss / (i + 1) (counts)
                    pbar.update(1)
            print(
                0,
                "{:02d} Val MegaMulAdd=".format(epoch),
                "time=",
                time.time() - start,
                "s",
            )
            m_iou, iou = evealer.getIoU()
            print("mean IOU", m_iou)
            logs[str(epoch)]["mIoU"] = m_iou
            tp, fp, fn = evealer.getStats()
            total = tp + fp + fn
            print("classes          IoU")
            print("----------------------------")
            for i in range(data.N_CLASSES):
                label_name = config["labels"][config["learning_map_inv"][i]]
                logs[str(epoch)][
                    "ValData"
                ] += "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})\n".format(
                    label_name, iou[i], tp[i], total[i]
                )
                print(
                    "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                        label_name, iou[i], tp[i], total[i]
                    )
                )
            json.dump(logs, open(logs_dir, "w"))
        sys.stdout.flush()
    skip_val =False