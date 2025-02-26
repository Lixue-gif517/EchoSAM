import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
import torch

# 欧式距离
def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

# 求pck指标，公式见我的博客
def pck_cal(p, g, N=2):
    # indices = torch.tensor([[0, 2, 4], [1, 3, 5]])
    # p = p.view(1,6)
    # g = g.view(1,6)
    # p = p[0, indices]
    # g = g[0, indices]
    # p是预测坐标，g是真值，N是总帧数
    set_value = 0.2
    result_1 = []
    result_2 = []
    result_3 = []
    result_all = []
    thresholds = np.linspace(0, 0.5, 100)
    p = p.cpu()
    g = g.cpu()
    p = p.numpy()
    g = g.numpy()
    for thr in thresholds:
        results = np.full((2, 3), 0, dtype=np.float32)
        for i in range(N):
            pi = p[i]
            p_ = np.split(pi, [2,4])
            gi = g[i]
            g_ = np.split(gi, [2, 4])
            for j in range(len(p_)):
                dd = np.linalg.norm(p_[j] - g_[j])
                d = distance.euclidean(p_[j], g_[j])
                # if thr < 0.21:
                #     if 0.7*d <= thr:
                #         results[i, j] = 1
                # else:
                if 1*d <= thr:
                    results[i, j] = 1
        mean_points = np.mean(results, axis=0)
        result_1.append(mean_points[0])
        result_2.append(mean_points[1])
        result_3.append(mean_points[2])
        result_all.append(np.mean(mean_points))
        cc = '%.1f' % thr
        if set_value == float('%.1f' % thr):
            aa = np.mean(mean_points)
        # show_pck_curve(thresholds, result_1)
        # show_pck_curve(thresholds, result_2)
        # show_pck_curve(thresholds, result_3)
    # show_pck_curve(thresholds, result_all)

    return aa,result_1,result_2,result_3,result_all


def show_pck_curve(thresholds,result):
    # 绘制PCK曲线
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)

    plt.plot(thresholds, result, label='PCK Curve', color='blue', linewidth=2)
    plt.title('PCK Curve')
    plt.xlabel('Threshold (Normalized Distance)')
    plt.ylabel('Accuracy')
    plt.axhline(y=0.5, color='red', linestyle='--', label='50% Accuracy Line')
    plt.axvline(x=0.2, color='green', linestyle='--', label='Typical Threshold')
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.xlim(0, 0.5)
    plt.tight_layout()
    plt.savefig('sine_wave.png')
    plt.show()


def compute_pck_pckh(dt_kpts,gt_kpts,refer_kpts):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)
    assert(len(refer_kpts)==2)
    assert(dt.shape[0]==gt.shape[0])
    ranges=np.arange(0.0,0.1,0.01)
    kpts_num=gt.shape[2]
    ped_num=gt.shape[0]
    #compute dist
    scale=np.sqrt(np.sum(np.square(gt[:,:,refer_kpts[0]]-gt[:,:,refer_kpts[1]]),1))
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    #compute pck
    pck = np.zeros([ranges.shape[0], gt.shape[2]+1])
    for idh,trh in enumerate(list(ranges)):
        for kpt_idx in range(kpts_num):
            pck[idh,kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= trh)
        # compute average pck
        pck[idh,-1] = 100*np.mean(dist <= trh)
    return pck

def pck_curve():
    import pandas as pd
    import seaborn as sns
    # df_pck = pd.read_csv('pck_csv.csv')
    df_pck = pd.read_csv('/home/lixue/Medseg/EchoSAM/pck_csv.csv')
    thresholds = np.linspace(0, 0.5, 100).tolist()
    fig = plt.figure(figsize=(18, 4))
    sns.set_style("darkgrid")
    with sns.axes_style("white"):
        plt.subplot(1, 4, 1)
        sns.lineplot(x="x", y="pall", data=df_pck, ci=95, label="IF+MF",linestyle='--')
        sns.lineplot(x="x", y="pall_", data=df_pck, ci=95, label="IF")
        # sns.lineplot(x="x", y="p1", data=df_pck, ci=None, label="p1")
        # sns.lineplot(x="x", y="p2", data=df_pck, ci=None, label="p2")
        # sns.lineplot(x="x", y="p3", data=df_pck, ci=None,  label="p3")
        # plt.xlabel("threshold")
        plt.title("Average")
        plt.ylabel("pck")
        plt.xlabel("T")

    with sns.axes_style("white"):
        plt.subplot(1, 4, 2)

        sns.lineplot(x="x", y="p1", data=df_pck, ci=95, label="IF+MF",linestyle='--')
        sns.lineplot(x="x", y="p1_", data=df_pck, ci=95, label="IF")

        # sns.lineplot(x="x", y="pall_", data=df_pck, ci=95, label="all",linestyle='--')
        # sns.lineplot(x="x", y="p1_", data=df_pck, ci=None, label="p1")
        # sns.lineplot(x="x", y="p2_", data=df_pck, ci=None, label="p2")
        # sns.lineplot(x="x", y="p3_", data=df_pck, ci=None,  label="p3")
        # plt.xlabel("threshold")
        plt.ylabel("pck")
        plt.xlabel("T")
        plt.title("Anterior leaflet annulus")

    with sns.axes_style("white"):
        plt.subplot(1, 4, 3)

        sns.lineplot(x="x", y="p2", data=df_pck, ci=95, label="IF+MF",linestyle='--')
        sns.lineplot(x="x", y="p2_", data=df_pck, ci=95, label="IF")

        # sns.lineplot(x="x", y="pall_", data=df_pck, ci=95, label="all",linestyle='--')
        # sns.lineplot(x="x", y="p1_", data=df_pck, ci=None, label="p1")
        # sns.lineplot(x="x", y="p2_", data=df_pck, ci=None, label="p2")
        # sns.lineplot(x="x", y="p3_", data=df_pck, ci=None,  label="p3")
        # plt.xlabel("threshold")
        plt.ylabel("pck")
        plt.title("Posterior leaflet annulus")
        plt.xlabel("T")

    with sns.axes_style("white"):
        plt.subplot(1, 4, 4)

        sns.lineplot(x="x", y="p3", data=df_pck, ci=95, label="IF+MF",linestyle='--')
        sns.lineplot(x="x", y="p3_", data=df_pck, ci=95, label="IF")

        # sns.lineplot(x="x", y="pall_", data=df_pck, ci=95, label="all",linestyle='--')
        # sns.lineplot(x="x", y="p1_", data=df_pck, ci=None, label="p1")
        # sns.lineplot(x="x", y="p2_", data=df_pck, ci=None, label="p2")
        # sns.lineplot(x="x", y="p3_", data=df_pck, ci=None,  label="p3")
        # plt.xlabel("threshold")
        plt.ylabel("pck")
        plt.title("Heart apex")
        plt.xlabel("T")

    plt.savefig("pck_if.png",dpi=400)
    plt.show()




if __name__ == '__main__':
    pck_curve()
    # np.random.seed(10)  # 这里改换随机种子，得到不同的pck
    # batch = 16
    # a = 0.7
    # # 这里只是随机数，之后可以换成自己需要的预测值和真值
    # pred = np.random.rand(batch, 6)
    # ground = np.random.rand(batch, 6)
    # out = pck(pred, ground, batch)
    # print('pck = ', out)
