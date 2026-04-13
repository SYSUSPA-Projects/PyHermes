import time
import pickle

import numpy as np
from mpi4py import MPI

from pyhermes.io import WindowFunc, ConvolsData, Corr3PCFMultipoleData
from pyhermes.utils import func_util, math_util
from pyhermes.pipeline import TaskBase


class Corr_3PCF_Multipole(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params(self):
        self.convols_data_path = self.task_params["convols_data_path"]
        self.convols_data1_path = self.task_params.get("convols_data1_path", "") or self.convols_data_path
        self.convols_data2_path = self.task_params.get("convols_data2_path", "") or self.convols_data_path
        self.convols_data3_path = self.task_params.get("convols_data3_path", "") or self.convols_data_path
        self.fout_path = self.task_params["fout_path"]

        win_params = self.task_params.get("window", None)
        win_params = win_params if (win_params and win_params.get("type")) else None

        for i in range(1, 4):
            win_params_i = self.task_params.get(f"window{i}", None)
            win_params_i = win_params_i if (win_params_i and win_params_i.get("type")) else None
            if (not win_params_i) and win_params:
                win_params_i = dict(win_params)
            setattr(self, f"win_params{i}", win_params_i)

        self.r12 = float(self.task_params.get("r12", self.task_params.get("r1")))
        self.r13 = float(self.task_params.get("r13", self.task_params.get("r2")))
        self.l_min = int(self.task_params["l_min"])
        self.l_max = int(self.task_params["l_max"])
        self.gpu_device_id = int(self.task_params["gpu_device_id"])
        self.field_mode = self.task_params["field_mode"]
        self.execution_mode = self.task_params["execution_mode"]
        self.cache_multipole_fields = bool(self.task_params["cache_multipole_fields"])
        self.cache_dir = self.task_params["cache_dir"]
        self.verbose_m_progress = bool(self.task_params["verbose_m_progress"])
        self.threads = int(self.task_params["threads"])

    def _spawn_windowed(self, base_convols, win_params):
        if win_params:
            window = WindowFunc(win_params, base_convols.convols_info, threads=self.threads)
            return base_convols @ window
        cdata = base_convols._spawn_like()
        cdata.epsilon = base_convols.epsilon
        cdata.format_convols_params()
        return cdata

    def _broadcast_convols(self, rank, comm, convols_data):
        serialized = pickle.dumps(convols_data.convols_info) if rank == 0 else None
        serialized = comm.bcast(serialized, root=0)
        if rank == 0:
            local = convols_data
            local.epsilon = np.ascontiguousarray(local.epsilon, dtype=np.float64)
        else:
            local = ConvolsData(threads=self.threads)
            local.convols_info = pickle.loads(serialized)
            local.format_convols_params()
            local.epsilon = np.empty((local.L, local.L, local.L), dtype=np.float64)
        comm.Bcast(local.epsilon, root=0)
        return local

    def _prepare_output(self, local_convols, l_arr, multipole_l):
        rho1 = 1.0 / local_convols[0].V
        rho2 = 1.0 / local_convols[1].V
        rho3 = 1.0 / local_convols[2].V
        self.corr3pcf_multipole_data.r12 = self.r12
        self.corr3pcf_multipole_data.r13 = self.r13
        self.corr3pcf_multipole_data.l = l_arr
        if self.field_mode == "raw":
            self.corr3pcf_multipole_data.ddd_l = multipole_l
            self.corr3pcf_multipole_data.delta_ddd_l = None
            self.corr3pcf_multipole_data.zeta_l = None
        else:
            self.corr3pcf_multipole_data.ddd_l = None
            self.corr3pcf_multipole_data.delta_ddd_l = multipole_l
            self.corr3pcf_multipole_data.zeta_l = multipole_l / (rho1 * rho2 * rho3)

    def _log_helpers(self):
        def _format_complex(value):
            real = float(value.real)
            imag = float(value.imag)
            if abs(imag) < 1e-12 * max(1.0, abs(real)):
                return f"{real:.5e}"
            return f"({real:.5e}, {imag:.5e})"

        def _log_l_progress(
            l,
            l_max,
            ddd_l,
            zeta_l,
            elapsed_sec,
            conv_elapsed_sec,
            sum_elapsed_sec,
            completed_m_tasks,
            total_m_tasks,
        ):
            progress = (completed_m_tasks / total_m_tasks) * 100.0
            if self.field_mode == "raw":
                stat_str = f"ddd_l={ddd_l:.5e}"
            else:
                stat_str = f"delta_ddd_l={ddd_l:.5e} | zeta_l={zeta_l:.5e}"
            self.logger.info(
                f" l={l:2d}/{l_max:2d} done | {stat_str} | "
                f"elapsed={elapsed_sec:.2f} sec | conv={conv_elapsed_sec:.2f} sec | "
                f"sum={sum_elapsed_sec:.2f} sec | progress={progress:6.2f}% "
                f"({completed_m_tasks}/{total_m_tasks} m-tasks)"
            )

        def _log_m_progress(l, l_max, m, m_max, value, elapsed_sec, completed_m_tasks, total_m_tasks):
            progress = (completed_m_tasks / total_m_tasks) * 100.0
            msg = (
                f"   m={m:2d}/{m_max:2d} in l={l:2d}/{l_max:2d} | "
                f"value={_format_complex(value)} | elapsed={elapsed_sec:.2f} sec | "
                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks})"
            )
            print(msg, flush=True)

        return _log_l_progress, _log_m_progress

    def _run_serial_mode(self, rank):
        if rank != 0:
            return

        self.logger.info("Start to calculate 3PCF multipole ...")
        self.logger.info(
            f"execution_mode={self.execution_mode}, field_mode={self.field_mode}, "
            f"l_min={self.l_min}, l_max={self.l_max}, threads={self.threads}, "
            f"cache_multipole_fields={self.cache_multipole_fields}, "
            f"verbose_m_progress={self.verbose_m_progress}"
        )
        rho1 = 1.0 / self.convols_data1.V
        rho2 = 1.0 / self.convols_data2.V
        rho3 = 1.0 / self.convols_data3.V
        if self.field_mode == "raw":
            field1 = self.convols_data1
            field2 = self.convols_data2
            field3 = self.convols_data3
        elif self.field_mode == "delta":
            field1 = self.convols_data1 - rho1
            field2 = self.convols_data2 - rho2
            field3 = self.convols_data3 - rho3
        else:
            self.logger.error(f"Unsupported field_mode='{self.field_mode}'. Use 'raw' or 'delta'.")
            func_util.safe_exit(1)

        log_l_progress, log_m_progress = self._log_helpers()
        l_arr, multipole_l, timing_info = math_util.calc_DDD_multipole(
            field1, field2, field3,
            self.r12, self.r13, self.l_min, self.l_max,
            gpu_device_id=self.gpu_device_id,
            cache_multipole_fields=self.cache_multipole_fields,
            cache_dir=self.cache_dir,
            threads=self.threads,
            progress_callback=log_l_progress if self.verbose_m_progress else None,
            m_progress_callback=log_m_progress if self.verbose_m_progress else None,
        )
        self._prepare_output([self.convols_data1, self.convols_data2, self.convols_data3], l_arr, multipole_l)
        if self.verbose_m_progress:
            self.logger.info(
                f"3PCF multipole timing | convolution={timing_info['conv_elapsed_sec']:.2f} sec | "
                f"summation={timing_info['sum_elapsed_sec']:.2f} sec"
            )
            self.logger.info(
                f"3PCF multipole summation breakdown | "
                f"h2d={timing_info['sum_h2d_elapsed_sec']:.2f} sec | "
                f"kernel={timing_info['sum_kernel_elapsed_sec']:.2f} sec | "
                f"d2h={timing_info['sum_d2h_elapsed_sec']:.2f} sec | "
                f"reduce={timing_info['sum_reduce_elapsed_sec']:.2f} sec | "
                f"callback={timing_info['sum_callback_elapsed_sec']:.2f} sec"
            )

    def _run_pair_mpi_mode(self, comm, rank, local_convols):
        size = comm.Get_size()
        if size == 1:
            if rank == 0:
                self.logger.warning(
                    "execution_mode='pair_mpi' requested with a single MPI rank. "
                    "Falling back to serial execution."
                )
                self.execution_mode = "serial"
                self._run_serial_mode(rank)
            return
        if size < 2 or size % 2 != 0:
            self.logger.error("execution_mode='pair_mpi' requires an even number of MPI ranks.")
            func_util.safe_exit(1)
        n_pairs = size // 2

        if rank == 0:
            self.logger.info("Start to calculate 3PCF multipole ...")
            self.logger.info(
                f"execution_mode={self.execution_mode}, field_mode={self.field_mode}, "
                f"l_min={self.l_min}, l_max={self.l_max}, threads={self.threads}, ranks={size}, pairs={n_pairs}, "
                f"cache_multipole_fields={self.cache_multipole_fields}, "
                f"verbose_m_progress={self.verbose_m_progress}"
            )

        rho1 = 1.0 / local_convols[0].V
        rho2 = 1.0 / local_convols[1].V
        rho3 = 1.0 / local_convols[2].V
        if self.field_mode == "raw":
            field1 = local_convols[0]
            field2 = local_convols[1]
            field3 = local_convols[2]
        elif self.field_mode == "delta":
            field1 = local_convols[0] - rho1
            field2 = local_convols[1] - rho2
            field3 = local_convols[2] - rho3
        else:
            self.logger.error(f"Unsupported field_mode='{self.field_mode}'. Use 'raw' or 'delta'.")
            func_util.safe_exit(1)

        pair_idx = rank if rank < n_pairs else rank - n_pairs
        is_r1_rank = rank < n_pairs

        conv_context_r1 = math_util._prepare_legendre_convolution_context(field2) if is_r1_rank else None
        conv_context_r2 = math_util._prepare_legendre_convolution_context(field3) if not is_r1_rank else None
        gpu_context = math_util._prepare_multipole_gpu_context(field1, gpu_device_id=self.gpu_device_id) if rank == 0 else None

        task_list = []
        l_arr = np.arange(self.l_min, self.l_max + 1, dtype=np.int32)
        for l_idx, l in enumerate(range(self.l_min, self.l_max + 1)):
            for m in range(0, l + 1):
                task_list.append((l_idx, l, m))
        multipole_l = np.empty(l_arr.size, dtype=np.float64) if rank == 0 else None
        m_storage = {int(l): np.empty(int(l) + 1, dtype=np.complex128) for l in l_arr} if rank == 0 else None
        done_per_l = {int(l): 0 for l in l_arr} if rank == 0 else None
        total_conv_elapsed = 0.0
        total_sum_elapsed = 0.0
        total_comm_elapsed = 0.0
        total_h2d = total_kernel = total_reduce = total_d2h = 0.0
        total_m_tasks = len(task_list)
        completed_m_tasks = 0
        _, log_m_progress = self._log_helpers()
        l_wall_starts = ({int(l): None for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)
        l_conv_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)
        l_comm_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)
        l_sum_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)

        n_rounds = (len(task_list) + n_pairs - 1) // n_pairs
        for round_idx in range(n_rounds):
            active_tasks = task_list[round_idx * n_pairs : (round_idx + 1) * n_pairs]
            active_count = len(active_tasks)
            round_meta = np.full((n_pairs, 3), -1, dtype=np.int32)
            for idx, (l_idx, l, m) in enumerate(active_tasks):
                round_meta[idx] = (l_idx, l, m)
            comm.Bcast(round_meta, root=0)

            t_conv = time.perf_counter()
            local_field = None
            local_meta = tuple(round_meta[pair_idx])
            if pair_idx < active_count:
                _, l, m = local_meta
                if is_r1_rank:
                    local_field = math_util._stream_convolution_fields(
                        field2, self.r12, int(l), threads=self.threads, m_values=[int(m)], conv_context=conv_context_r1
                    )[0]
                else:
                    local_field = math_util._stream_convolution_fields(
                        field3, self.r13, int(l), threads=self.threads, m_values=[-int(m)], conv_context=conv_context_r2
                    )[0]
            conv_elapsed = time.perf_counter() - t_conv
            total_conv_elapsed += conv_elapsed

            t_comm = time.perf_counter()
            if pair_idx < active_count:
                l_idx, l, m = local_meta
                tag_base = 200000 + round_idx * 100 + pair_idx
                if rank == 0:
                    round_fields = {}
                    if pair_idx == 0:
                        round_fields[(int(l_idx), int(l), int(m))] = [local_field, None]
                    for idx in range(active_count):
                        recv_l_idx, recv_l, recv_m = map(int, round_meta[idx])
                        key = (recv_l_idx, recv_l, recv_m)
                        if idx == 0:
                            recv_r2 = np.empty(local_field.shape, dtype=np.complex128)
                            comm.Recv([recv_r2, MPI.COMPLEX16], source=n_pairs, tag=tag_base + 50)
                            round_fields[key][1] = recv_r2
                        else:
                            recv_r1 = np.empty(local_field.shape, dtype=np.complex128)
                            recv_r2 = np.empty(local_field.shape, dtype=np.complex128)
                            comm.Recv([recv_r1, MPI.COMPLEX16], source=idx, tag=200000 + round_idx * 100 + idx)
                            comm.Recv([recv_r2, MPI.COMPLEX16], source=n_pairs + idx, tag=200000 + round_idx * 100 + idx + 50)
                            round_fields[key] = [recv_r1, recv_r2]
                elif is_r1_rank:
                    if rank != 0:
                        comm.Send([np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16], dest=0, tag=tag_base)
                else:
                    comm.Send([np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16], dest=0, tag=tag_base + 50)
            comm_elapsed = time.perf_counter() - t_comm
            total_comm_elapsed += comm_elapsed

            round_timings = None
            if self.verbose_m_progress:
                local_timing = np.array(
                    [
                        int(pair_idx if pair_idx < active_count else -1),
                        int(0 if is_r1_rank else 1),
                        float(conv_elapsed),
                        float(comm_elapsed),
                    ],
                    dtype=np.float64,
                )
                round_timings = comm.gather(local_timing, root=0)

            if rank == 0:
                timing_by_task = {}
                if self.verbose_m_progress:
                    for item in round_timings:
                        idx_float, side_float, conv_val, comm_val = item
                        idx_task = int(idx_float)
                        if idx_task < 0 or idx_task >= active_count:
                            continue
                        side = int(side_float)
                        if idx_task not in timing_by_task:
                            timing_by_task[idx_task] = {}
                        timing_by_task[idx_task][side] = (float(conv_val), float(comm_val))
                for idx in range(active_count):
                    l_idx, l, m = map(int, round_meta[idx])
                    key = (l_idx, l, m)
                    field_r1_m, field_r2_m = round_fields[key]
                    if self.verbose_m_progress and l_wall_starts[l] is None:
                        l_wall_starts[l] = time.perf_counter()
                    task_timing = timing_by_task.get(idx, {})
                    conv_r1 = task_timing.get(0, (0.0, 0.0))[0]
                    conv_r2 = task_timing.get(1, (0.0, 0.0))[0]
                    comm_r1 = task_timing.get(0, (0.0, 0.0))[1]
                    comm_r2 = task_timing.get(1, (0.0, 0.0))[1]
                    t_sum = time.perf_counter()
                    value, timing = math_util.compute_multipole_m_summand(field_r1_m, field_r2_m, gpu_context)
                    sum_elapsed = time.perf_counter() - t_sum
                    total_sum_elapsed += sum_elapsed
                    total_h2d += timing["h2d_elapsed_sec"]
                    total_kernel += timing["kernel_elapsed_sec"]
                    total_reduce += timing["reduce_elapsed_sec"]
                    total_d2h += timing["d2h_elapsed_sec"]
                    m_storage[l][m] = value
                    done_per_l[l] += 1
                    completed_m_tasks += 1
                    if self.verbose_m_progress:
                        l_conv_accum[l] += max(conv_r1, conv_r2)
                        l_comm_accum[l] += max(comm_r1, comm_r2)
                        l_sum_accum[l] += sum_elapsed
                    if self.verbose_m_progress:
                        log_m_progress(
                            l=l, l_max=self.l_max, m=m, m_max=l, value=value,
                            elapsed_sec=max(conv_r1, conv_r2) + max(comm_r1, comm_r2) + sum_elapsed,
                            completed_m_tasks=completed_m_tasks, total_m_tasks=total_m_tasks,
                        )
                    if done_per_l[l] == l + 1:
                        multipole_l[l_idx] = math_util.combine_multipole_m_terms(m_storage[l], l)
                        zeta_l = multipole_l[l_idx] / (rho1 * rho2 * rho3)
                        progress = (completed_m_tasks / total_m_tasks) * 100.0
                        if self.field_mode == "raw":
                            stat_str = f"ddd_l={multipole_l[l_idx]:.5e}"
                        else:
                            stat_str = f"delta_ddd_l={multipole_l[l_idx]:.5e} | zeta_l={zeta_l:.5e}"
                        if self.verbose_m_progress:
                            self.logger.info(
                                f" l={l:2d}/{self.l_max:2d} done | {stat_str} | "
                                f"elapsed={time.perf_counter() - l_wall_starts[l]:.2f} sec | "
                                f"conv={l_conv_accum[l]:.2f} sec | comm={l_comm_accum[l]:.2f} sec | "
                                f"sum={l_sum_accum[l]:.2f} sec | "
                                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks} m-tasks)"
                            )
                        else:
                            self.logger.info(
                                f" l={l:2d}/{self.l_max:2d} done | {stat_str} | "
                                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks} m-tasks)"
                            )

        conv_sum_all = comm.reduce(total_conv_elapsed, op=MPI.SUM, root=0) if self.verbose_m_progress else None
        conv_max_rank = comm.reduce(total_conv_elapsed, op=MPI.MAX, root=0) if self.verbose_m_progress else None
        comm_sum_all = comm.reduce(total_comm_elapsed, op=MPI.SUM, root=0) if self.verbose_m_progress else None
        comm_max_rank = comm.reduce(total_comm_elapsed, op=MPI.MAX, root=0) if self.verbose_m_progress else None
        if rank == 0:
            self._prepare_output(local_convols, l_arr, multipole_l)
            if self.verbose_m_progress:
                self.logger.info(
                    f"Pair-MPI timing | conv_rank0={total_conv_elapsed:.2f} sec | conv_sum_all={conv_sum_all:.2f} sec | "
                    f"conv_max_rank={conv_max_rank:.2f} sec | comm_sum_all={comm_sum_all:.2f} sec | "
                    f"comm_max_rank={comm_max_rank:.2f} sec | summation={total_sum_elapsed:.2f} sec"
                )
                self.logger.info(
                    f"Pair-MPI summation breakdown | h2d={total_h2d:.2f} sec | kernel={total_kernel:.2f} sec | "
                    f"d2h={total_d2h:.2f} sec | reduce={total_reduce:.2f} sec"
                )

    def run(self, convols_data1=None, convols_data2=None, convols_data3=None, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            if rank == 0:
                t0 = time.perf_counter()

            self.format_params()
            self.corr3pcf_multipole_data = Corr3PCFMultipoleData()
            convols_info1_serialized = None
            convols_info2_serialized = None
            convols_info3_serialized = None

            if rank == 0:
                base_convols_cache = {}

                def load_base_convols(path):
                    if path not in base_convols_cache:
                        self.logger.info(f"Initializing multipole input on rank 0: loading base ConvolsData from {path} ...")
                        base_convols_cache[path] = ConvolsData(data_path=path, threads=self.threads)
                    return base_convols_cache[path]

                if not (
                    (convols_data1 is not None or self.convols_data1_path)
                    and (convols_data2 is not None or self.convols_data2_path)
                    and (convols_data3 is not None or self.convols_data3_path)
                ):
                    self.logger.error(
                        "Missing input field(s). Please either pass convols_data1/2/3 or specify "
                        "'convols_data_path' / 'convols_data1_path' / 'convols_data2_path' / "
                        "'convols_data3_path' in task_params."
                    )
                    func_util.safe_exit(1)

                for i, cdata in zip([1, 2, 3], [convols_data1, convols_data2, convols_data3]):
                    if cdata is not None:
                        if isinstance(cdata, ConvolsData):
                            self.logger.info(
                                f"Initializing multipole input on rank 0: preparing field leg {i} | "
                                "provided ConvolsData, no additional window convolution"
                            )
                            _convols_data = cdata
                            setattr(self, f"convols_data{i}", cdata)
                        else:
                            self.logger.error(f"convols_data{i} is not ConvolsData.")
                            func_util.safe_exit(1)
                    else:
                        _base_path = getattr(self, f"convols_data{i}_path")
                        _base_convols = load_base_convols(_base_path)
                        self.logger.info(
                            f"Initializing multipole input on rank 0: preparing field leg {i} | "
                            f"{func_util.describe_window_action(getattr(self, f'win_params{i}', None))}"
                        )
                        _convols_data = self._spawn_windowed(_base_convols, getattr(self, f"win_params{i}", None))
                        setattr(self, f"convols_data{i}", _convols_data)

                    setattr(self.corr3pcf_multipole_data, f"convols_info{i}", getattr(self, f"convols_data{i}").convols_info)

                self.corr3pcf_multipole_data.corr3pcf_multipole_info = dict(self.task_params)
                convols_info1_serialized = pickle.dumps(self.convols_data1.convols_info)
                convols_info2_serialized = pickle.dumps(self.convols_data2.convols_info)
                convols_info3_serialized = pickle.dumps(self.convols_data3.convols_info)

            convols_info1_serialized = comm.bcast(convols_info1_serialized, root=0)
            convols_info2_serialized = comm.bcast(convols_info2_serialized, root=0)
            convols_info3_serialized = comm.bcast(convols_info3_serialized, root=0)

            if rank == 0:
                t_setup_done = time.perf_counter()
                self.logger.info(f"Pre-3PCF multipole setup time: {t_setup_done - t0:.4f} sec")

            if self.execution_mode == "pair_mpi":
                if rank == 0:
                    self.logger.info(
                        f"Initializing multipole input: broadcasting smoothed fields to {comm.Get_size()} MPI ranks ..."
                    )
                local_convols = [
                    self._broadcast_convols(rank, comm, self.convols_data1 if rank == 0 else None),
                    self._broadcast_convols(rank, comm, self.convols_data2 if rank == 0 else None),
                    self._broadcast_convols(rank, comm, self.convols_data3 if rank == 0 else None),
                ]
                if rank == 0:
                    self.logger.info("Initializing multipole input: broadcast complete, entering MPI convolution stage ...")
                self._run_pair_mpi_mode(comm, rank, local_convols)
            else:
                if rank == 0:
                    local_convols = [self.convols_data1, self.convols_data2, self.convols_data3]
                else:
                    local_convols = []
                    for serialized in [convols_info1_serialized, convols_info2_serialized, convols_info3_serialized]:
                        cdata = ConvolsData(threads=self.threads)
                        cdata.convols_info = pickle.loads(serialized)
                        cdata.format_convols_params()
                        local_convols.append(cdata)
                self._run_serial_mode(rank)

            if rank == 0 and self.fout_path:
                self.corr3pcf_multipole_data.saveflag = True
                self.corr3pcf_multipole_data.save_corr3pcf_multipole(self.fout_path, overwrite=overwrite)

            comm.Barrier()
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

        if self.rank == 0:
            t1 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {t1 - t0:.4f} sec")

        return self.corr3pcf_multipole_data
