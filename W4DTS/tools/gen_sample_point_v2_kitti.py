import os, glob, shutil, yaml, json, time, sys,random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import pairwise_distances

learn_map = yaml.safe_load(
    open(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset",
            "config",
            "semantic-kitti.yaml",
        )
    )
)

resolution = 3
base_dir = "/path/to/dataset"
data_dir = os.path.join(base_dir,'sequences')
sv_dir = os.path.join(base_dir,'superV_v2')
output_dir = os.path.join(base_dir,'sv_sample_point_v3')

os.makedirs(output_dir, exist_ok=True)
for i in range(20):
    os.makedirs(os.path.join(output_dir, "{:02d}".format(i)), exist_ok=True)

d_format = ""
sv_format = "{}/superV/{}.npy"
tmp_format = "{} {} {} {:d} {:d} {:d}\n"
o_json_format = "{}/{}.json"


def mid_point(points, o_idx, sp, label_s):
    clusters = np.unique(sp, axis=0, return_index=True, return_inverse=True)
    potential_id = [
        (i, c)
        for i, c in enumerate(clusters[0])
        if np.sum(clusters[2] == i) >= min([5, (len(o_idx) * 0.1 + 1)])
    ]
    if len(potential_id) > 1:
        potential_id =  random.sample(potential_id,1)
    elif not len(potential_id):
        potential_id = [(i, c) for i, c in enumerate(clusters[0])]
        potential_id = random.sample(potential_id,1)
    potential_id = potential_id[0][1]
    return int(potential_id)


def getGreedyPerm(D,rstart):
    N = D.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[rstart, :]
    perm[0] = rstart
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def fd_sample(points,clusters,potential_id,sm):
    mean_list = []
    for i, c in potential_id:
        mean = points[clusters[2]==i].mean(0).reshape(-1,3)
        mean_list.append(mean)
    rstart = random.randint(0,len(mean_list))
    mean_list = np.vstack(mean_list)
    D = pairwise_distances(mean_list, metric='euclidean')
    sample = getGreedyPerm(D,rstart)
    ms = sample[0][:sm]
    return [x for i,x in enumerate(potential_id) if i in ms]


def pv_point(points, o_idx, sp, label_s):
    label_s_lm = np.zeros_like(label_s)
    for k, v in learn_map["learning_map"].items():
        label_s_lm[label_s == k] = v

    l, ri = np.unique(label_s_lm, return_inverse=True)
    ms = []

    for i, v in enumerate(l):
        if not v:
            continue
        pts = ri == i
        # t_idx = o_idx[pts]
        t_sp = sp[pts]
        t_points = points[pts]

        clusters = np.unique(t_sp, axis=0, return_index=True, return_inverse=True)
        potential_id = [
            (i, c)
            for i, c in enumerate(clusters[0])
            if np.sum(clusters[2] == i) >= min([5, (len(o_idx) * 0.1 + 1)])
        ]
        # if v in [9,10,11,12,13,15,]:
        #     sm = min([20, len(potential_id)//20])
        # else:
        #     sm = 1
        sm = min([20, len(potential_id)])

        if not len(potential_id):
            potential_id = [(i, c) for i, c in enumerate(clusters[0])]
        sm = min([20, len(potential_id)])
        if len(potential_id) > sm:
            # potential_id = random.sample(potential_id,sm)
            potential_id = fd_sample(t_points,clusters,potential_id,sm)
        potential_id = [int(x[1]) for x in potential_id]
        ms.extend(potential_id)
    return ms


def supervoxel_gen(data):
    s_id = data.split("/")[-3]
    f_id = data.split("/")[-1].split(".")[0]
    # if not os.path.exists(data.replace("velodyne", "superV").replace(".bin", ".npy")):
    #     return
    scan = np.fromfile(
        data,
        dtype=np.float32,
    ).reshape(-1, 4)[:,:3]
    label = np.fromfile(
        data.replace("velodyne", "labels").replace(".bin", ".label"), dtype=np.uint32
    )
    sp = np.load(
        open(data.replace(data_dir, sv_dir).replace("velodyne", "superV").replace(".bin", ".npy"), "rb")
    )
    # scan, sp = scan[:, :3], scan[:, 3:]
    label_s = label & 0xFFFF
    label_o = label >> 16
    p_dict = {}
    for oidx in np.unique(label_o):
        p_index = label_o == oidx  # if  oidx >0 else  label_o == oidx
        op_idx = np.where(label_o == oidx)[0]
        if oidx > 0:
            p_dict[int(oidx)] = mid_point(
                scan[p_index], op_idx, sp[p_index], label_s[p_index]
            )

        else:
            p_dict[int(oidx)] = pv_point(
                scan[p_index], op_idx, sp[p_index], label_s[p_index]
            )

    json.dump(
        p_dict, open(os.path.join(output_dir, o_json_format.format(s_id, f_id)), "w")
    )


if __name__ == "__main__":

    data_list = glob.glob(os.path.join(data_dir, "*", "velodyne", "*"))
    data_list = [x for x in data_list if int(x.split("/")[-3]) <= 10]

    executor = ThreadPoolExecutor(max_workers=8)

    while data_list:
        if not os.path.exists(
            data_list[0].replace(data_dir, sv_dir).replace("velodyne", "superV").replace(".bin", ".npy")
        ):
            sys.stdout.write("wait for new: {} left\n".format(len(data_list)))
            sys.stdout.flush()
            time.sleep(600)
            continue
        data = data_list[0]
        # supervoxel_gen(data)
        executor.submit(supervoxel_gen,data)
        data_list.pop(0)
        if not len(data_list) % 128:
            sys.stdout.write("now: {} left\n".format(len(data_list)))
            sys.stdout.flush()
            executor.shutdown(wait=True)
            executor = ThreadPoolExecutor(max_workers=8)
        break
    sys.stdout.write("finish!\n")
    sys.stdout.flush()
