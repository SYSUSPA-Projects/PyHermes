import pickle
import sys
import time

import numpy as np
import pywt
from numba import cuda

print(cuda.list_devices())
device_id = 0
cuda.select_device(device_id)
cur_device = cuda.get_current_device()
cur_device.reset()


def cal_gamma(phi_data, PhiSupport, SampRate):
    Gamma = np.zeros((PhiSupport, PhiSupport))
    for l1 in range(PhiSupport):
        for l2 in range(PhiSupport):
            rolled_phi1 = np.roll(phi_data, l1 * SampRate)
            rolled_phi2 = np.roll(phi_data, l2 * SampRate)
            Gamma[l1, l2] = np.sum(phi_data * rolled_phi1 * rolled_phi2) / SampRate
    return Gamma


@cuda.jit
def compute_3d_result_gpu(data, data_R20, data_R40, Gamma, result, L, PhiSupport):
    lx, ly, lz = cuda.grid(3)

    if lx < L and ly < L and lz < L:
        sum_over_l1 = 0
        for l1x in range(PhiSupport):
            index_l1x = (lx - l1x) % L
            for l1y in range(PhiSupport):
                index_l1y = (ly - l1y) % L
                for l1z in range(PhiSupport):
                    index_l1z = (lz - l1z) % L

                    sum_over_l2 = 0
                    for l2x in range(PhiSupport):
                        index_l2x = (lx - l2x) % L
                        res_y = 0
                        for l2y in range(PhiSupport):
                            index_l2y = (ly - l2y) % L
                            res_z = 0
                            for l2z in range(PhiSupport):
                                index_l2z = (lz - l2z) % L
                                res_z += Gamma[l1z, l2z] * data_R40[index_l2x, index_l2y, index_l2z]
                            res_y += Gamma[l1y, l2y] * res_z
                        sum_over_l2 += Gamma[l1x, l2x] * res_y

                    sum_over_l1 += data_R20[index_l1x, index_l1y, index_l1z] * sum_over_l2

        result[lx, ly, lz] = data[lx, ly, lz] * sum_over_l1


J = 8
L = 1 << J

wavelet = pywt.Wavelet("db2")
phi, psi, x = wavelet.wavefun(level=10)
phi_data = phi[:-1]
Radius_all = [15]
PhiSupport = 3
SampRate = 1024
Gamma = cal_gamma(phi_data, PhiSupport, SampRate)

print("Dot products matrix:")
print(Gamma)
# R_vector = [20, 40]

## load the data
time_all_start = time.perf_counter()
for Radius in Radius_all:
    work_dir = "/home/tristan/graduate/association_hermes/mdpl/data/r" + str(Radius)
    all_l_result = []
    with open(
        work_dir + "/deltac_" + str(L) + "_005_r" + str(Radius) + "_pywt.pk",
        "rb",
    ) as f:
        data = pickle.load(f)
    Gamma_gpu = cuda.to_device(Gamma)
    result_gpu = cuda.device_array((L, L, L), dtype=np.complex128)
    data_size = 283116474
    rho = data_size / L**2
    data_gpu = cuda.to_device(data)
    R1 = 30
    R2 = 60

    for l in range(8):
        l_result = []
        # l = sys.argv[-1]
        # l = int(sys.argv[1])

        for m in range(l + 1):
            if m == 0:
                with open(
                    work_dir
                    + "/R"
                    + str(R1)
                    + "/deltac_"
                    + str(L)
                    + "_005_r"
                    + str(Radius)
                    + "_R"
                    + str(R1)
                    + "_l"
                    + str(l)
                    + "_m0_pywt.pk",
                    "rb",
                ) as f:
                    data_R20 = pickle.load(f)
                with open(
                    work_dir
                    + "/R"
                    + str(R2)
                    + "/deltac_"
                    + str(L)
                    + "_005_r"
                    + str(Radius)
                    + "_R"
                    + str(R2)
                    + "_l"
                    + str(l)
                    + "_m0_pywt.pk",
                    "rb",
                ) as f:
                    data_R40 = pickle.load(f)
            elif m > 0:
                with open(
                    work_dir
                    + "/R"
                    + str(R1)
                    + "/deltac_"
                    + str(L)
                    + "_005_r"
                    + str(Radius)
                    + "_R"
                    + str(R1)
                    + "_l"
                    + str(l)
                    + "_m"
                    + str(m)
                    + "_pywt.pk",
                    "rb",
                ) as f:
                    data_R20 = pickle.load(f)
                with open(
                    work_dir
                    + "/R"
                    + str(R2)
                    + "/deltac_"
                    + str(L)
                    + "_005_r"
                    + str(Radius)
                    + "_R"
                    + str(R2)
                    + "_l"
                    + str(l)
                    + "_m_minus"
                    + str(m)
                    + "_pywt.pk",
                    "rb",
                ) as f:
                    data_R40 = pickle.load(f)
            time_load_start = time.perf_counter()
            data_R20_gpu = cuda.to_device(data_R20)
            data_R40_gpu = cuda.to_device(data_R40)
            time_load_end = time.perf_counter()
            time_load_cost_gpu = time_load_end - time_load_start
            # print("GPU Time for load data: ", time_load_cost_gpu)

            threads_per_block = (8, 8, 8)
            blocks_per_grid = (
                (L + threads_per_block[0] - 1) // threads_per_block[0],
                (L + threads_per_block[1] - 1) // threads_per_block[1],
                (L + threads_per_block[2] - 1) // threads_per_block[2],
            )
            # compute_3d_result_gpu[blocks_per_grid, threads_per_block](
            # data_gpu, data_R20_gpu, data_R40_gpu, Gamma_gpu, result_gpu, L, PhiSupport
            # )
            # cuda.synchronize()
            time_start = time.perf_counter()
            compute_3d_result_gpu[blocks_per_grid, threads_per_block](
                data_gpu, data_R20_gpu, data_R40_gpu, Gamma_gpu, result_gpu, L, PhiSupport
            )
            cuda.synchronize()
            time_end = time.perf_counter()
            time_cost_gpu = time_end - time_start
            print("GPU Time single cost in seconds: ", time_cost_gpu)
            result = result_gpu.copy_to_host()

            result_sum = np.sum(result) / rho**3
            print("l = :", str(l), " m = :", str(m), " Result sum: ", result_sum * 4 * np.pi)
            l_result.append(result_sum * 4 * np.pi)
        all_l_result.append(l_result)
    with open("./data/l_result_" + str(Radius) + "_R" + str(R1) + "_" + str(R2) + ".pk", "wb") as f:
        pickle.dump(all_l_result, f)
