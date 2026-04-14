import time
import pickle
import copy
import numpy as np

from pyhermes.io import WindowFunc, ConvolsData, Corr3PCFData
from pyhermes.utils import func_util, math_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.pipeline import TaskBase

from .corr2pcf import calc_DD_mean_r


def calc_DDD_mean_mc(
    r12_scaled, r13_scaled, theta, pos_scaled, n_rot,
    convols_meta, convols_data2, convols_data3,
    center="random",
    seed_base_rot=-1,
    theta_index=-1,
    eps1=None,
    rho1=None,
):
    """
    Wrapper around numba kernels:
    - center="random"   -> calc_DDD_mc_random_center
    - center="particle" -> calc_DDD_mc_pos_center_fast

    All lengths/positions are SCALED (GRID) units here.
    """
    kwargs_common = {
        "phi_data": convols_meta.phi_data,
        "L": convols_meta.L,
        "SampRate": convols_meta.SampRate,
        "PhiSupport": convols_meta.PhiSupport,
        "seed_base_rot": seed_base_rot,
        "theta_index": theta_index
    }

    eps2 = convols_data2.epsilon
    eps3 = convols_data3.epsilon

    if center == "random":
        if eps1 is None:
            raise ValueError("eps1 must be provided when center='random'.")
        res = math_util.calc_DDD_mc_random_center(
            r12_scaled, r13_scaled, theta,
            pos_scaled, n_rot,
            eps1, eps2, eps3,
            **kwargs_common,
        )
    elif center == "particle":
        if rho1 is None:
            raise ValueError("rho1 must be provided when center='particle'.")
        res = math_util.calc_DDD_mc_pos_center_fast(
            r12_scaled, r13_scaled, theta,
            pos_scaled, n_rot,
            rho1, eps2, eps3,
            **kwargs_common,
        )
    else:
        raise ValueError(f"Unknown center='{center}'. Use 'random' or 'particle'.")

    return res


class Corr_3PCF(TaskBase):
    """
    3PCF task.

    Parallelization:
      - Each MPI rank works on a subset of centers (pos_local).
      - Every rank computes all theta values.
      - Results are combined by weighted averaging with weights = N_local centers.

    Random strategy:
      - centers ("random"): each rank generates its own centers using a rank-dependent seed
      - rotations: handled inside math_util.calc_DDD_mc_* using deterministic seeds
        based on (theta_index, rot_index), identical across ranks.
    """

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_3PCF": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.convols_data1 = None
        self.convols_data2 = None
        self.convols_data3 = None
        self.window1 = None
        self.window2 = None
        self.window3 = None
        self._fields_prepared = False

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params["threads"] = self.threads
        self.sync_runtime_options(context="Corr_3PCF runtime configuration", blank_line=True)

    def format_params(self):
        self.convols_data_path = self.task_params["convols_data_path"]
        self.convols_data1_path = self.task_params.get("convols_data1_path", "") or self.convols_data_path
        self.convols_data2_path = self.task_params.get("convols_data2_path", "") or self.convols_data_path
        self.convols_data3_path = self.task_params.get("convols_data3_path", "") or self.convols_data_path
        self.fout_path = self.task_params["fout_path"]
        self.threads = int(self.task_params["threads"])

        win_params = self.task_params.get("window", None)
        win_params = win_params if (win_params and win_params.get("type")) else None

        for i in range(1, 4):
            win_params_i = self.task_params.get(f"window{i}", None)
            win_params_i = win_params_i if (win_params_i and win_params_i.get("type")) else None
            if (not win_params_i) and win_params:
                win_params_i = dict(win_params)
            setattr(self, f"win_params{i}", win_params_i)

        self.r12 = float(self.task_params["r12"])   # physical (Mpc/h)
        self.r13 = float(self.task_params["r13"])   # physical (Mpc/h)

        self.theta_min = 0.0
        self.theta_max = np.pi
        self.n_theta = int(self.task_params["n_theta"])
        self.n_rot = int(self.task_params["n_rot"])

        self.center = self.task_params["center"]      # "random" or "particle"
        self.field_mode = self.task_params["field_mode"]  # "raw" or "delta"
        self.n_rand = int(self.task_params["n_rand"]) # total centers when center="random"
        self.base_seed = int(self.task_params["base_seed"])

    def _current_task_params_snapshot(self):
        params = copy.deepcopy(self.task_params)
        params["convols_data_path"] = self.convols_data_path
        params["convols_data1_path"] = self.convols_data1_path
        params["convols_data2_path"] = self.convols_data2_path
        params["convols_data3_path"] = self.convols_data3_path
        params["fout_path"] = self.fout_path
        params["threads"] = self.threads
        params["r12"] = self.r12
        params["r13"] = self.r13
        params["n_theta"] = self.n_theta
        params["n_rot"] = self.n_rot
        params["center"] = self.center
        params["field_mode"] = self.field_mode
        params["n_rand"] = self.n_rand
        params["base_seed"] = self.base_seed
        return params

    def _resolve_base_convols(self, leg_idx, provided_convols, base_convols_cache):
        if provided_convols is not None:
            if not isinstance(provided_convols, ConvolsData):
                self.logger.error(f"Unexpected input: 'convols_data{leg_idx}' is not an instance of 'ConvolsData'.")
                func_util.safe_exit(1)
            return provided_convols, f"provided convols_data{leg_idx}"

        base_path = getattr(self, f"convols_data{leg_idx}_path")
        if not base_path:
            self.logger.error(
                f"Missing input for field leg {leg_idx}. Please pass convols_data{leg_idx} or set "
                f"'convols_data{leg_idx}_path' / 'convols_data_path'."
            )
            func_util.safe_exit(1)
        if base_path not in base_convols_cache:
            base_convols_cache[base_path] = ConvolsData(data_path=base_path, threads=self.threads)
        return base_convols_cache[base_path], f"path={base_path}"

    def _resolve_window(self, leg_idx, base_convols, provided_window):
        if isinstance(provided_window, WindowFunc):
            return provided_window, "provided WindowFunc instance"
        if isinstance(provided_window, dict):
            return WindowFunc(provided_window, base_convols.convols_info, threads=self.threads), (
                f"provided window dict | {func_util.describe_window_action(provided_window)}"
            )
        if provided_window is not None:
            self.logger.error(
                f"Unsupported window input for leg {leg_idx}. Expected dict, WindowFunc, or None, "
                f"got {type(provided_window)}."
            )
            func_util.safe_exit(1)

        config_window = getattr(self, f"win_params{leg_idx}", None)
        if config_window:
            return WindowFunc(config_window, base_convols.convols_info, threads=self.threads), (
                f"config window{leg_idx} | {func_util.describe_window_action(config_window)}"
            )
        return None, "no additional window convolution"

    def prepare_input_fields(
        self,
        convols_data1=None,
        convols_data2=None,
        convols_data3=None,
        window1=None,
        window2=None,
        window3=None,
    ):
        self.corr3pcf_data = Corr3PCFData()
        self._sync_runtime_options()
        if convols_data1 is None:
            convols_data1 = self.convols_data1
        if convols_data2 is None:
            convols_data2 = self.convols_data2
        if convols_data3 is None:
            convols_data3 = self.convols_data3
        if window1 is None:
            window1 = self.window1
        if window2 is None:
            window2 = self.window2
        if window3 is None:
            window3 = self.window3

        if self.rank == 0:
            self.logger.info("Preparing Corr_3PCF input fields ...")
            self.logger.info(
                f"field_mode={self.field_mode}, center={self.center}, "
                f"n_theta={self.n_theta}, n_rot={self.n_rot}, "
                f"r12={self.r12}, r13={self.r13}"
            )
            base_convols_cache = {}
            resolved_legs = []
            for i, cdata, win in zip(
                [1, 2, 3],
                [convols_data1, convols_data2, convols_data3],
                [window1, window2, window3],
            ):
                base_convols, source_desc = self._resolve_base_convols(i, cdata, base_convols_cache)
                resolved_legs.append((i, base_convols, source_desc, win))

            shared_required = func_util.validate_convols_compatibility(
                [item[1] for item in resolved_legs],
                ConvolsData._REQUIRED_ARGV,
                logger=self.logger,
                label="Corr_3PCF input fields",
            )
            shared_required_text = ", ".join([f"{k}={v}" for k, v in shared_required.items()])
            self.logger.info("Corr_3PCF input compatibility check passed.")
            self.logger.info(f"Shared required parameters | {shared_required_text}")

            for i, base_convols, source_desc, win in resolved_legs:
                window_obj, window_desc = self._resolve_window(i, base_convols, win)
                if window_obj is not None:
                    final_convols = base_convols @ window_obj
                else:
                    final_convols = base_convols.copy()
                    final_convols.format_convols_params()

                setattr(self, f"convols_data{i}", final_convols)
                setattr(self, f"window{i}", window_obj)
                setattr(self.corr3pcf_data, f"convols_info{i}", final_convols.convols_info)
                self.logger.info(
                    f"Field leg {i} ready | source={source_desc} | window={window_desc}"
                )

            self.corr3pcf_data.corr3pcf_info = self._current_task_params_snapshot()
            self.corr3pcf_data.task_params = self._current_task_params_snapshot()
        self._fields_prepared = True

    def run(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()

            if rank == 0:
                t0 = time.perf_counter()

            if not self._fields_prepared:
                self.prepare_input_fields()
            base_seed = self.base_seed
            center = self.center
            n_rand = self.n_rand
            field_mode = self.field_mode

            # -------------------------------
            # Load / build convols_data1/2/3 on rank 0, broadcast convols_info, Bcast epsilon
            # -------------------------------
            convols_info1_serialized = None
            convols_info2_serialized = None
            convols_info3_serialized = None

            if rank == 0:
                convols_info1_serialized = pickle.dumps(self.convols_data1.convols_info)
                convols_info2_serialized = pickle.dumps(self.convols_data2.convols_info)
                convols_info3_serialized = pickle.dumps(self.convols_data3.convols_info)

            convols_info1_serialized = comm.bcast(convols_info1_serialized, root=0)
            convols_info2_serialized = comm.bcast(convols_info2_serialized, root=0)
            convols_info3_serialized = comm.bcast(convols_info3_serialized, root=0)

            if rank == 0:
                if center == "random":
                    self.convols_data1.epsilon = np.ascontiguousarray(self.convols_data1.epsilon, dtype=np.float64)
                self.convols_data2.epsilon = np.ascontiguousarray(self.convols_data2.epsilon, dtype=np.float64)
                self.convols_data3.epsilon = np.ascontiguousarray(self.convols_data3.epsilon, dtype=np.float64)
                _local_convols1 = self.convols_data1
                _local_convols2 = self.convols_data2
                _local_convols3 = self.convols_data3
            else:
                _local_convols1 = ConvolsData(threads=self.threads)
                _local_convols1.convols_info = pickle.loads(convols_info1_serialized)
                _local_convols1.format_convols_params()
                if center == "random":
                    _local_convols1.epsilon = np.empty((_local_convols1.L, _local_convols1.L, _local_convols1.L), dtype=np.float64)
                else:
                    _local_convols1.epsilon = None

                _local_convols2 = ConvolsData(threads=self.threads)
                _local_convols2.convols_info = pickle.loads(convols_info2_serialized)
                _local_convols2.format_convols_params()
                _local_convols2.epsilon = np.empty((_local_convols2.L, _local_convols2.L, _local_convols2.L), dtype=np.float64)

                _local_convols3 = ConvolsData(threads=self.threads)
                _local_convols3.convols_info = pickle.loads(convols_info3_serialized)
                _local_convols3.format_convols_params()
                _local_convols3.epsilon = np.empty((_local_convols3.L, _local_convols3.L, _local_convols3.L), dtype=np.float64)

            self.corr3pcf_data.corr3pcf_info = self._current_task_params_snapshot()
            self.corr3pcf_data.task_params = self._current_task_params_snapshot()

            if center == "random":
                comm.Bcast(_local_convols1.epsilon, root=0)
            comm.Bcast(_local_convols2.epsilon, root=0)
            comm.Bcast(_local_convols3.epsilon, root=0)
            comm.Barrier()

            # -------------------------------
            # Prepare theta and seeds (all ranks compute all theta)
            # -------------------------------
            if rank == 0:
                self.logger.info("Start to calculate 3PCF (pos-parallel) ...")
                if center == "random":
                    self.logger.info(
                        f"center={center}, field_mode={field_mode}, n_rot={self.n_rot}, "
                        f"n_theta={self.n_theta}, n_rand(total)={n_rand}"
                    )
                elif center == "particle":
                    pos_all = _local_convols1.get_particle_data() * _local_convols1.ScaleFactor  # (N,3)
                    Nall = pos_all.shape[0]
                    self.logger.info(
                        f"center={center}, field_mode={field_mode}, n_rot={self.n_rot}, "
                        f"n_theta={self.n_theta}, n_particle(total)={Nall}"
                    )
                else:
                    raise ValueError(f"Unknown center='{center}'. Use 'random' or 'particle'.")
                theta_arr = np.linspace(self.theta_min, self.theta_max, self.n_theta)
            else:
                theta_arr = None
            theta_arr = comm.bcast(theta_arr, root=0)

            # seed_base_rot: used inside numba kernels to create per-(theta,rot) seeds
            seed_base_rot = base_seed + 1

            # -------------------------------
            # Build pos_local (POS-parallel)
            # -------------------------------
            if center == "random":
                # each rank gets a local count
                if rank == 0:
                    counts = np.full(size, n_rand // size, dtype=np.int64)
                    counts[: (n_rand % size)] += 1
                else:
                    counts = None

                n_local = int(comm.scatter(counts, root=0))

                # rank-dependent seed ensures different center sets across ranks
                seed_center_rank = base_seed + 1000003 * (rank + 1)
                pos_local = math_util.random_points_box(N=n_local, SimBoxL=_local_convols1.L, seed=seed_center_rank)

            elif center == "particle":
                # Scatterv particle centers (scaled to GRID units)
                if rank == 0:
                    counts = np.full(size, Nall // size, dtype=np.int64)
                    counts[: (Nall % size)] += 1
                    displs = np.zeros(size, dtype=np.int64)
                    displs[1:] = np.cumsum(counts[:-1])

                    pos_all = np.ascontiguousarray(pos_all, dtype=np.float64)
                    sendbuf = pos_all.ravel()
                    counts3 = counts * 3
                    displs3 = displs * 3
                else:
                    sendbuf = None
                    counts = None
                    counts3 = None
                    displs3 = None

                n_local = int(comm.scatter(counts, root=0))
                recvbuf = np.empty(n_local * 3, dtype=np.float64)
                comm.Scatterv([sendbuf, counts3, displs3, MPI.DOUBLE], recvbuf, root=0)
                pos_local = recvbuf.reshape(n_local, 3)

            else:
                raise ValueError(f"Unknown center='{center}'. Use 'random' or 'particle'.")

            npos_local = pos_local.shape[0]
            npos_total = comm.allreduce(npos_local, op=MPI.SUM)

            if rank == 0:
                self.logger.info(f"Total centers used: {npos_total} (distributed over {size} ranks)")
                t_start = time.perf_counter()
                self.logger.info(f"Pre-3PCF setup time: {t_start - t0:.4f} sec")

            # -------------------------------
            # Progress reporting (pos-parallel): track per-rank theta completion
            # -------------------------------
            total_tasks = len(theta_arr)
            local_completed = 0

            # -------------------------------
            # Main loop over theta on each rank (local means)
            # -------------------------------
            rho1 = 1.0 / _local_convols1.V
            rho2 = 1.0 / _local_convols2.V
            rho3 = 1.0 / _local_convols3.V
            RRR = rho1 * rho2 * rho3
            if field_mode == "raw":
                field_convols1 = _local_convols1
                field_convols2 = _local_convols2
                field_convols3 = _local_convols3
            elif field_mode == "delta":
                field_convols1 = None if center == "particle" else (_local_convols1 - rho1)
                field_convols2 = _local_convols2 - rho2
                field_convols3 = _local_convols3 - rho3
            else:
                raise ValueError(f"Unknown field_mode='{field_mode}'. Use 'raw' or 'delta'.")
            r12_scaled = self.r12 * _local_convols1.ScaleFactor
            r13_scaled = self.r13 * _local_convols1.ScaleFactor

            if rank == 0:
                t_ddd_start = time.perf_counter()

            local_DDD_mean = np.empty(theta_arr.shape[0], dtype=np.float64)

            for it, th in enumerate(theta_arr):
                theta_start = time.perf_counter()
                local_DDD_mean[it] = calc_DDD_mean_mc(
                    r12_scaled, r13_scaled, th,
                    pos_local, self.n_rot,
                    _local_convols1, field_convols2, field_convols3,
                    center=center,
                    seed_base_rot=seed_base_rot,
                    theta_index=it,
                    eps1=None if field_convols1 is None else field_convols1.epsilon,
                    rho1=rho1,
                )
                local_completed += 1
                if rank == 0:
                    theta_elapsed = time.perf_counter() - theta_start
                    self.logger.info(
                        f" theta[{it + 1:02d}/{total_tasks:02d}] done | "
                        f"theta={th:.5f} rad | local_DDD_mean={local_DDD_mean[it]:.5e} | "
                        f"elapsed={theta_elapsed:.2f} sec"
                    )

            # weighted reduce to global mean
            local_weighted = local_DDD_mean * npos_local
            global_weighted = np.empty_like(local_weighted)
            comm.Allreduce(local_weighted, global_weighted, op=MPI.SUM)

            if rank == 0:
                t_ddd_end = time.perf_counter()
                self.logger.info(f"DDD main loop time: {t_ddd_end - t_ddd_start:.4f} sec")
                self.logger.info("Main DDD loop finished, computing xi12/xi13/xi23 on rank 0 ...")
                DDD_mean_global = global_weighted / npos_total
                self.corr3pcf_data.theta = theta_arr
                self.corr3pcf_data.r23 = math_util.third_side(self.r12, self.r13, theta_arr)
                r23_chunks = np.array_split(self.corr3pcf_data.r23, size)
            else:
                DDD_mean_global = None
                r23_chunks = None

            # -------------------------------
            # Compute xi12/xi13 on rank 0 and xi23 in parallel across ranks
            # -------------------------------
            RR12 = rho1 * rho2
            RR13 = rho1 * rho3
            RR23 = rho2 * rho3
            delta_convols1 = None
            if rank == 0:
                delta_convols1 = field_convols1 if field_convols1 is not None else (_local_convols1 - rho1)
            delta_convols2 = field_convols2
            delta_convols3 = field_convols3

            if rank == 0:
                t_xi12 = time.perf_counter()
                xi12 = calc_DD_mean_r(self.r12, delta_convols1, delta_convols2) / RR12
                t_xi12 = time.perf_counter() - t_xi12

                t_xi13 = time.perf_counter()
                xi13 = calc_DD_mean_r(self.r13, delta_convols1, delta_convols3) / RR13
                t_xi13 = time.perf_counter() - t_xi13
            else:
                xi12 = None
                xi13 = None
                t_xi12 = None
                t_xi13 = None

            t_xi23_start = time.perf_counter()
            r23_local = comm.scatter(r23_chunks, root=0)
            if len(r23_local) > 0:
                xi23_local = np.array([calc_DD_mean_r(r, delta_convols2, delta_convols3) / RR23 for r in r23_local], dtype=np.float64)
            else:
                xi23_local = np.empty(0, dtype=np.float64)
            gathered_xi23 = comm.gather(xi23_local, root=0)

            if rank == 0:
                xi23 = np.concatenate(gathered_xi23) if gathered_xi23 else np.empty(0, dtype=np.float64)
                t_xi23 = time.perf_counter() - t_xi23_start

                self.logger.info(
                    f"Post-processing timing | xi12={t_xi12:.2f} sec | "
                    f"xi13={t_xi13:.2f} sec | xi23={t_xi23:.2f} sec"
                )

                if field_mode == "raw":
                    xi12 -= 1.0
                    xi13 -= 1.0
                    xi23 -= 1.0

                self.corr3pcf_data.xi12 = xi12
                self.corr3pcf_data.xi13 = xi13
                self.corr3pcf_data.xi23 = xi23
                self.corr3pcf_data.ddd = None
                self.corr3pcf_data.delta_ddd = None
                # pdelta_ddd stores <D_1 (D_2-R) (D_3-R)> for particle-centered delta runs.
                self.corr3pcf_data.pdelta_ddd = None

                if field_mode == "raw":
                    zeta_p = DDD_mean_global / RRR - 1.0
                    self.corr3pcf_data.ddd = DDD_mean_global
                    self.corr3pcf_data.zeta = zeta_p - xi12 - xi13 - xi23
                elif center == "random":
                    self.corr3pcf_data.delta_ddd = DDD_mean_global
                    self.corr3pcf_data.zeta = DDD_mean_global / RRR
                else:
                    self.corr3pcf_data.pdelta_ddd = DDD_mean_global
                    self.corr3pcf_data.zeta = DDD_mean_global / RRR - xi23

                self.corr3pcf_data.Q = self.corr3pcf_data.zeta / (xi12 * xi23 + xi13 * xi23 + xi12 * xi13)

                t_end = time.perf_counter()
                self.logger.info(f"The time for 3PCF (pos-parallel): {t_end - t_start:.4f} sec")

                # Save
                if save_result and self.fout_path:
                    self.logger.info("Saving 3PCF result to output file ...")
                    self.corr3pcf_data.saveflag = True
                    self.corr3pcf_data.save_corr3pcf(self.fout_path, overwrite=overwrite)

            comm.Barrier()

        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

        if self.rank == 0:
            t1 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {t1 - t0:.4f} sec")

        return self.corr3pcf_data
