
import json

elastic_deformation = False
import MinkowskiEngine as ME

# print(ME.__file__)
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage
from W4DTS.utils.generate_sequential import *
from MinkowskiEngine.utils import sparse_quantize, sparse_collate
from W4DTS.dataset.dataset_mf import BaseDataset
from  sklearn.neighbors import KDTree

config = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "config/semantic-kitti.yaml"))  # -part
)
N_CLASSES = len(config["learning_map_inv"])


class SmanticKITTI(BaseDataset):
    mid_data_format = "{}/{}.npy"

    def __init__(
        self,
        data_base,
        batch_size,
        window_size,
        scale,
        mid_label,
        config=config,
        type="train",
        sv_dir=None,
        sample_interval=None,
        sample_dir='sv_sample_point_v2',
        kp = 1
    ):
        if sample_interval:
            self.sample_interval = sample_interval
        else:
            self.sample_interval = window_size
        self.sample_dir = sample_dir
        self.kp = kp
        super(SmanticKITTI, self).__init__(
            data_base,
            batch_size,
            window_size,
            scale,
            mid_label,
            type,
            config,
            sv_dir=sv_dir,
        )


    def get_files(self, type):
        input_base = os.path.join(self.data_base, "sequences")
        dirs = glob.glob(os.path.join(input_base, "*"))

        for dir_ in dirs:
            sq = int(dir_[-2:])
            split = [k for k, v in self.config["split"].items() if sq in v][0]
            if not split == self.type:
                continue
            datas = glob.glob(os.path.join(dir_, "velodyne", "*"))
            labels = glob.glob(os.path.join(dir_, "labels", "*"))
            if self.sv_dir:
                sv_masks = glob.glob(
                    os.path.join(self.sv_dir, dir_[-2:], "superV", "*")
                )
            else:
                sv_masks = glob.glob(os.path.join(dir_, "superV", "*"))
            clib = parse_calibration(os.path.join(dir_, "calib.txt"))
            poses = parse_poses(os.path.join(dir_, "poses.txt"), clib)

            smaple_indexs = glob.glob(
                os.path.join(self.data_base, self.sample_dir, dir_[-2:], "*")
            )
            datas = list(sorted(datas, key=lambda x: int(x[-10:-4])))
            labels = list(sorted(labels, key=lambda x: int(x[-12:-6])))
            sv_masks = list(sorted(sv_masks, key=lambda x: int(x[-10:-4])))
            smaple_indexs = list(sorted(smaple_indexs, key=lambda x: int(x[-11:-5])))

            f = []
            datas = datas
            poses = poses
            for i, data in enumerate(datas):
                pre = (
                    i
                    if i + self.window_size < len(datas)
                    else len(datas) - self.window_size
                )
                data_frames = datas[pre : pre + self.window_size]
                data_frames = [data_frames[0]] + data_frames
                pose_frames = poses[pre : pre + self.window_size]
                pose_frames = [pose_frames[0]] + pose_frames
                if i % self.sample_interval:
                    smaple_index = ['']*(self.window_size+1)
                else:
                    smaple_index = [smaple_indexs[pre]]*2 + ['']*(self.window_size-1)
                label_frames = labels[pre : pre + self.window_size]
                label_frames = [label_frames[0]] + label_frames
                sv_frames = sv_masks[pre : pre + self.window_size]
                sv_frames = [sv_frames[0]] + sv_frames

                f.append(
                    (
                        sq,
                        data_frames,
                        label_frames,
                        smaple_index,
                        pose_frames,
                        sv_frames,
                    )
                )
            self.files.extend(f)

    def aug(self, a, seed):
        a = np.matmul(a, seed)
        return a[:, :3]

    def __getitem__(self, idx):
        # pos = idx#self.files[idx]
        locs = []
        feats = []
        labels_pre = []
        labels_all = []
        poses = []
        reverse = []
        svs = []
        mid_sv = []
        o_coords = []
        sv_coords_indexes = []

        if self.type == "train":
            seed = np.eye(3) + np.random.randn(3, 3) * 0.01
            seed[0][0] *= np.random.randint(0, 2) * 2 - 1
            # seed *= shift_scale #scale
            theta = np.random.rand() * 2 * math.pi
            seed = np.matmul(
                seed,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )
        file_list = self.files[idx]
        for f in range(self.window_size + 1):
            poses.append(file_list[4][f])

            scan = np.fromfile(file_list[1][f], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            coords = scan[:, :3]
            r = scan[:, 3] - np.min(scan[:, 3])  # np.ones_like(scan[:, 3] )#
            if self.type == "train":
                coords = self.aug(coords, seed)
            coords = coords  # - coords.min(0)
            # o_coord = np.copy(coords)
            o_coords.append(torch.Tensor(np.copy(coords)))
            if file_list[3][f]:
                sample_point = json.load(open(file_list[3][f]))
                p_idx = sample_point["0"] + [v for k, v in sample_point.items() if k != "0"]
            else:
                p_idx=[]

            sv = np.load(open(file_list[5][f], "rb"))
            sv_info = list(np.unique(sv, return_inverse=True, return_counts=True))

            sv_coord = (
                torch.zeros([sv_info[0].shape[0], 3]).index_add_(
                    0, torch.IntTensor(sv_info[1]).long(), torch.Tensor(coords)
                )
            ) / torch.Tensor(sv_info[2]).view(-1, 1)
            if self.kp>1:
                sv_coords_indexes.append(
                    KDTree(sv_coord.numpy(), leaf_size=10).query(sv_coord.numpy(),k=self.kp,return_distance=False)
                )#int(sv_coord.shape[0] * 0.6)
            else:
                sv_coords_indexes.append(
                    None
                )

            sv_info.append(p_idx)
            svs.append(sv_info)

            coords /= self.scale
            coords = coords.astype(np.int32)

            seq_idx = file_list[1][f].split("/")[-3]
            f_idx = file_list[1][f].split("/")[-1].split(".")[0]
            mid_tmp = os.path.join(
                self.mid_label,
                self.mid_data_format.format(
                    seq_idx, f_idx
                ),
            )

            label = np.fromfile(file_list[2][f], dtype=np.uint32)
            sem_label = label & 0xFFFF
            for k, v in self.config["learning_map"].items():
                sem_label[sem_label == k] = v
            labels_all.append(torch.Tensor(sem_label.astype(np.int)).view(-1, 1))
            sem_label_t = sem_label
            if f in [0,1] or not (f-1)%self.sample_interval:
                sv_mask = np.isin(sv, p_idx)
                sem_label[(1 - sv_mask).astype(np.bool)] = 0
                for p_i in p_idx:
                    p_lab = np.unique(sem_label[sv == p_i], return_counts=True)
                    if len(p_lab[0]) == 1:
                        continue
                    sem_label[sv == p_i] = p_lab[0][np.argmax(p_lab[1])]
                # sem_label = sem_label.astype(np.int)
            else:
                sem_label = np.zeros_like(sem_label)


            if os.path.isfile(mid_tmp):
                mid_sem = np.load(open(mid_tmp,'rb'))
                for sv_i,p_i in enumerate(mid_sem[:,0]):
                    sem_label[sv == p_i] =  mid_sem[sv_i,1]
                mid_sv.append(mid_sem)
            else:
                mid_sv.append(np.array([[-1,-1]]))
            sem_label = sem_label.astype(np.int)

            ind = sparse_quantize(
                torch.Tensor(coords),
                features=torch.Tensor(r.reshape(-1, 1)),
                ignore_label=0,
                return_index=True,
                return_inverse=True,
            )
            coords, r = ind[0], ind[1]

            coords, r = sparse_collate([coords], [r])
            locs.append(coords)
            feats.append(r.view(-1, 1))
            labels_pre.append(torch.Tensor(sem_label).view(-1, 1))
            reverse.append(ind[3])

        return {
            "idx": idx,
            "x": [locs, feats],
            "y": labels_pre,
            "y_all": labels_all,
            "reverse_index": reverse,
            "poses": poses,
            "svs": svs,
            "mid_sv": mid_sv,
            "o_coords": o_coords,
            "sv_coords_indexes": sv_coords_indexes,
        }

    def get_data_loader(self,):
        if self.type == "train":
            return torch.utils.data.DataLoader(
                self,
                batch_size=self.batch_size,
                collate_fn=lambda x: self.train_consist(x),
                num_workers=5,
                shuffle=True,
            )
        else:
            return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=5, shuffle=False, collate_fn=lambda x: self.train_consist(x),
            )

    def train_consist(self, x):
        tmp = {}
        # tmp['x']
        for idx, d in enumerate(x):
            if not idx:
                tmp["x"] = d["x"]
                tmp["idx"] = [d["idx"]]
                tmp["min_c"] = [min_c.min(0)[0].view(-1, 4) for min_c in d["x"][0]]
                for j in range(self.window_size + 1):
                    tmp["x"][0][j] -= tmp["min_c"][j]
                tmp["y"] = d["y"]
                tmp["y_all"] = d["y_all"]
                tmp["poses"] = d["poses"]
                tmp["poses"] = d["poses"]
                tmp["reverse_index"] = [d["reverse_index"]]
                tmp["svs"] = [d["svs"]]
                tmp["mid_sv"] = [d["mid_sv"]]
                tmp["o_coords"] = [d["o_coords"]]
                tmp["sv_coords_indexes"] = [d["sv_coords_indexes"]]
                # tmp["poses"][0] = np.eye(4)
            else:
                tmp["idx"].append(d["idx"])

                for j in range(self.window_size + 1):
                    tmp_x = d["x"][0][j]
                    tmp_x[:, 0] += idx
                    tmp["min_c"][j] = torch.cat(
                        (tmp["min_c"][j], d["x"][0][j].min(0)[0].view(-1, 4)),
                    ).view(-1, 4)
                    tmp_x[:, 1:] -= d["x"][0][j].min(0)[0][1:]
                    tmp["x"][0][j] = torch.cat((tmp["x"][0][j], tmp_x), 0)
                    tmp["x"][1][j] = torch.cat((tmp["x"][1][j], d["x"][1][j]), 0)
                    # if j:
                    tmp["y"][j] = torch.cat((tmp["y"][j], d["y"][j]), 0)
                    tmp["y_all"][j] = torch.cat((tmp["y_all"][j], d["y_all"][j]), 0)
                    tmp["poses"][j] = np.append(tmp["poses"][j], d["poses"][j]).reshape(
                        -1, 4, 4
                    )

                tmp["reverse_index"].append(d["reverse_index"])
                tmp["svs"].append(d["svs"])
                tmp["mid_sv"].append(d["mid_sv"])
                tmp["o_coords"].append(d["o_coords"])
                tmp["sv_coords_indexes"].append(d["sv_coords_indexes"])

        tmp["offset"] = [
            tmp["min_c"][si] - tmp["min_c"][si - 1] if si else tmp["min_c"][si]
            for si in range(len(tmp["min_c"]))
        ]
        return tmp