#include "htp-cpu-impl.h"
#include "htp-cpu-vec.h"

#include "htp-cpu-ops.h"
#include "htp-cpu-vec.h"
#include "htp-ops.h"
#include "../ggml-cpu/traits.h"
#include "../ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-threading.h"
#include "ggml.h"


// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    // if (!ggml_is_numa()) {
    //     return;
    // }

    // int node_num;
    // int rv;
    // size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    // switch(g_state.numa.numa_strategy) {
    //     case GGML_NUMA_STRATEGY_DISTRIBUTE:
    //         // run thread on node_num thread_n / (threads per node)
    //         node_num = thread_n % g_state.numa.n_nodes;
    //         break;
    //     case GGML_NUMA_STRATEGY_ISOLATE:
    //         // run thread on current_node
    //         node_num = g_state.numa.current_node;
    //         break;
    //     case GGML_NUMA_STRATEGY_NUMACTL:
    //         // use the cpuset that numactl gave us
    //         rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
    //         if (rv) {
    //             fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
    //         }
    //         return;
    //     default:
    //         return;
    // }

    // struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    // cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    // CPU_ZERO_S(setsize, cpus);
    // for (size_t i = 0; i < node->n_cpus; ++i) {
    //     CPU_SET_S(node->cpus[i], setsize, cpus);
    // }

    // rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    // if (rv) {
    //         fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    // }

    // CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    // if (!ggml_is_numa()) {
    //     return;
    // }

    // size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    // cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    // CPU_ZERO_S(setsize, cpus);
    // for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
    //     CPU_SET_S(i, setsize, cpus);
    // }

    // int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    // if (rv) {
    //     fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    // }

    // CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif


static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__) || (defined(__linux__) && defined(__ANDROID__))
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    // fprintf(stderr, "%s: set affinity mask val 0x%016llx\n", __func__, *(const unsigned long long *) mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

static void htp_threadpool_free(struct htp_threadpool * threadpool) {
    if (!threadpool) {
        return;
    }

    const int n_threads = threadpool->n_threads_max;

    struct htp_compute_state * workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);

    const size_t workers_size = sizeof(struct htp_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct htp_threadpool));
}

// pause/resume must be called under mutex
static void htp_threadpool_pause_locked(struct htp_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void htp_threadpool_resume_locked(struct htp_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}

static void htp_threadpool_pause(struct htp_threadpool * threadpool) {
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       htp_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
}

static void htp_threadpool_resume(struct htp_threadpool * threadpool) {
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       htp_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
}

static bool ggml_debug_tensor_stats(const struct ggml_tensor * t, double * min_v, double * max_v, double * mean_v) {
    if (!t || !t->data) {
        return false;
    }

    const int64_t ne = ggml_nelements(t);
    if (ne <= 0 || ne > INT_MAX) {
        return false;
    }

    double min_val = DBL_MAX;
    double max_val = -DBL_MAX;
    double sum_val = 0.0;

    for (int i = 0; i < (int) ne; ++i) {
        const double v = (double) ggml_get_f32_1d(t, i);
        min_val = MIN(min_val, v);
        max_val = MAX(max_val, v);
        sum_val += v;
    }

    *min_v  = min_val;
    *max_v  = max_val;
    *mean_v = sum_val / (double) ne;
    return true;
}

static bool ggml_debug_is_named_op(const struct ggml_tensor * t, enum ggml_op op, const char * name) {
    return t != NULL && t->op == op && name != NULL && t->name[0] != '\0' && strcmp(t->name, name) == 0;
}

enum ggml_debug_ffn_trace_kind {
    GGML_DEBUG_FFN_TRACE_NONE = 0,
    GGML_DEBUG_FFN_TRACE_GATE,
    GGML_DEBUG_FFN_TRACE_UP,
    GGML_DEBUG_FFN_TRACE_SWIGLU,
};

static enum ggml_debug_ffn_trace_kind ggml_debug_get_ffn_trace_kind_once(
        const struct htp_compute_state * state,
        const struct ggml_tensor * node) {
    static bool printed_gate = false;
    static bool printed_up = false;
    static bool printed_swiglu = false;

    if (state == NULL || state->ith != 0 || node == NULL) {
        return GGML_DEBUG_FFN_TRACE_NONE;
    }

    if (!printed_gate && ggml_debug_is_named_op(node, GGML_OP_MUL_MAT, "ffn_gate-0")) {
        printed_gate = true;
        return GGML_DEBUG_FFN_TRACE_GATE;
    }

    if (!printed_up && ggml_debug_is_named_op(node, GGML_OP_MUL_MAT, "ffn_up-0")) {
        printed_up = true;
        return GGML_DEBUG_FFN_TRACE_UP;
    }

    if (!printed_swiglu &&
            ggml_debug_is_named_op(node, GGML_OP_GLU, "ffn_swiglu-0") &&
            ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU) {
        printed_swiglu = true;
        return GGML_DEBUG_FFN_TRACE_SWIGLU;
    }

    return GGML_DEBUG_FFN_TRACE_NONE;
}

static void ggml_debug_dump_tensor_stats(const char * tag, const struct ggml_tensor * t) {
    double min_val = 0.0;
    double max_val = 0.0;
    double mean_val = 0.0;

    if (tag == NULL || t == NULL) {
        return;
    }

    if (ggml_debug_tensor_stats(t, &min_val, &max_val, &mean_val)) {
        fprintf(stderr,
                "HTP SWIGLU NODE STATS: %s name=%s op=%s type=%s ne=[%ld,%ld,%ld,%ld] min=%g max=%g mean=%g\n",
                tag,
                t->name[0] != '\0' ? t->name : "(unnamed)",
                ggml_op_name(t->op),
                ggml_type_name(t->type),
                t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                min_val, max_val, mean_val);
    } else {
        fprintf(stderr,
                "HTP SWIGLU NODE STATS: %s name=%s op=%s type=%s (stats unavailable)\n",
                tag,
                t->name[0] != '\0' ? t->name : "(unnamed)",
                ggml_op_name(t->op),
                ggml_type_name(t->type));
    }
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    bool enable_htp_profile = getenv("HTP_PROFILE") != NULL;
    bool enable_mul_mat_stats = getenv("HTP_DEBUG_MUL_MAT_STATS") != NULL;

    struct htp_compute_state * state = (struct htp_compute_state *) data;
    struct htp_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    int64_t t0 = ggml_time_us();
    int64_t npu_us = 0;
    int64_t cpu_us = 0;

    for (int node_n = 0; node_n < cgraph->n_nodes && !tp->abort; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        const enum ggml_debug_ffn_trace_kind debug_ffn_trace = ggml_debug_get_ffn_trace_kind_once(state, node);

        // if (state->ith == 0) {
        //     fprintf(stderr, "preparing to compute node %d %s\n", node_n, node->name);
        // }

        // if (state->ith == 0) {
        //     prepare_tensor_rpcmem_mapping(node);
        // }

        int64_t t1 = ggml_time_us();

        if (debug_ffn_trace == GGML_DEBUG_FFN_TRACE_SWIGLU) {
            fprintf(stderr,
                    "HTP SWIGLU NODE DEBUG: BEFORE node=%s op=%s glu_op=%d src0=%s src1=%s\n",
                    node->name[0] != '\0' ? node->name : "(unnamed)",
                    ggml_op_name(node->op),
                    (int) ggml_get_glu_op(node),
                    node->src[0] && node->src[0]->name[0] != '\0' ? node->src[0]->name : "(unnamed)",
                    node->src[1] && node->src[1]->name[0] != '\0' ? node->src[1]->name : "(null)");
            ggml_debug_dump_tensor_stats("ffn_swiglu.src0 BEFORE", node->src[0]);
            ggml_debug_dump_tensor_stats("ffn_swiglu.src1 BEFORE", node->src[1]);
        }

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort = true;
            tp->ec    = GGML_STATUS_ABORTED;
        }

        ggml_barrier_htp(state->threadpool);

        // if (state->ith == 0) {
        //     uint64_t d = * (uint64_t * )node->data;
        //     fprintf(stderr, "node %d %s, op %s, val %016lx\n", node_n, node->name, ggml_op_name(node->op), d);
        // }

        int64_t elapsed_time = ggml_time_us() - t1;
        const bool runs_on_npu = htp_ops_support_op(node);

        if (debug_ffn_trace == GGML_DEBUG_FFN_TRACE_GATE || debug_ffn_trace == GGML_DEBUG_FFN_TRACE_UP) {
            fprintf(stderr,
                    "HTP FFN CHAIN DEBUG: AFTER node=%s op=%s path=%s elapsed_us=%ld\n",
                    node->name[0] != '\0' ? node->name : "(unnamed)",
                    ggml_op_name(node->op),
                    runs_on_npu ? "NPU" : "CPU",
                    elapsed_time);
            if (debug_ffn_trace == GGML_DEBUG_FFN_TRACE_GATE) {
                ggml_debug_dump_tensor_stats("ffn_gate.dst AFTER", node);
            } else {
                ggml_debug_dump_tensor_stats("ffn_up.dst AFTER", node);
            }
        }

        if (debug_ffn_trace == GGML_DEBUG_FFN_TRACE_SWIGLU) {
            fprintf(stderr,
                    "HTP SWIGLU NODE DEBUG: AFTER node=%s op=%s path=%s elapsed_us=%ld\n",
                    node->name[0] != '\0' ? node->name : "(unnamed)",
                    ggml_op_name(node->op),
                    runs_on_npu ? "NPU" : "CPU",
                    elapsed_time);
            ggml_debug_dump_tensor_stats("ffn_swiglu.dst AFTER", node);
        }

        if (enable_mul_mat_stats && state->ith == 0 && node->op == GGML_OP_MUL_MAT) {
            double min_val = 0.0;
            double max_val = 0.0;
            double mean_val = 0.0;
            if (ggml_debug_tensor_stats(node, &min_val, &max_val, &mean_val)) {
                fprintf(stderr,
                        "HTP MUL_MAT STATS: node=%s path=%s dst_type=%s ne=[%ld,%ld,%ld,%ld] min=%g max=%g mean=%g\n",
                        node->name, runs_on_npu ? "NPU" : "CPU", ggml_type_name(node->type),
                        node->ne[0], node->ne[1], node->ne[2], node->ne[3], min_val, max_val, mean_val);
            } else {
                fprintf(stderr,
                        "HTP MUL_MAT STATS: node=%s path=%s dst_type=%s (stats unavailable)\n",
                        node->name, runs_on_npu ? "NPU" : "CPU", ggml_type_name(node->type));
            }
        }

        if (enable_htp_profile && state->ith == 0) {
            fprintf(stderr, "node %s, op %s, shape (%ld, %ld, %ld, %ld), %ld us",
                    node->name, ggml_op_name(node->op), node->ne[0], node->ne[1], node->ne[2], node->ne[3], elapsed_time);
            if (runs_on_npu) {
                fprintf(stderr, " on NPU\n");
                npu_us += elapsed_time;
            } else {
                fprintf(stderr, " on CPU (dst %s, src0 %s, src1 %s)\n",
                        ggml_type_name(node->type),
                        ggml_type_name(node->src[0]->type),
                        node->src[1] ? ggml_type_name(node->src[1]->type) : "--");
                cpu_us += elapsed_time;
            }
        }
    }

    int64_t elapsed_us = ggml_time_us() - t0;
    if (enable_htp_profile && state->ith == 0) {
        fprintf(stderr, "HTP: total %ld us, CPU %ld us, NPU %ld us\n", elapsed_us, cpu_us, npu_us);
    }
    return 0;
}

// check if thread is active
static inline bool ggml_graph_compute_thread_active(struct htp_compute_state * state) {
    struct htp_threadpool * threadpool = state->threadpool;
    int n_threads = atomic_load_explicit(&threadpool->n_threads_cur, memory_order_relaxed);
    return (state->ith < n_threads);
}

// check if thread is ready to proceed (exit from polling or sleeping)
static inline bool ggml_graph_compute_thread_ready(struct htp_compute_state * state) {
    struct htp_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int new_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    if (new_graph != state->last_graph) {
        state->pending    = ggml_graph_compute_thread_active(state);
        state->last_graph = new_graph;
    }

    return state->pending;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct htp_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct htp_compute_state * state) {
    struct htp_threadpool * threadpool = state->threadpool;

    // Skip polling for unused threads
    if (!ggml_graph_compute_thread_active(state)) {
        return state->pending;
    }

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct htp_compute_state * state) {
    struct htp_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct htp_compute_state * state = (struct htp_compute_state *) data;
    struct htp_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) {
            break;
        }

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;

            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct htp_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    GGML_PRINT_DEBUG("threadpool: n_threads_cur %d n_threads %d\n", threadpool->n_threads_cur, n_threads);

    // Update the number of active threads
    atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_fetch_add_explicit(&threadpool->n_graph, 1, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       htp_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

static struct htp_threadpool * htp_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct htp_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct htp_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = false;
        threadpool->workers          = NULL;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct htp_compute_state) * tpp->n_threads;
    struct htp_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }

    return threadpool;
}

// NOTE(hzx): This is a specialized version. Don't use ggml_cpu_init()
static void ggml_htp_cpu_init(void) {
    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_FP16_TO_FP32(u.fp16);
                ggml_table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

        is_first_call = false;
    }

    ggml_critical_section_end();
}

enum ggml_status ggml_graph_compute_htp_hybrid(struct ggml_cgraph *cgraph, struct ggml_cplan *cplan);

enum ggml_status ggml_graph_compute_htp_hybrid(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_htp_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct htp_threadpool * threadpool = (struct htp_threadpool *) cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);

        // NOTE(hzx): we may set ttp.cpumask & ttp.strict_cpu here for thread affinity.
        // ttp.cpumask[2] = 1;
        // ttp.cpumask[3] = 1;
        // ttp.cpumask[4] = 1;
        // ttp.cpumask[5] = 1;
        // ttp.strict_cpu = 0;

        threadpool = htp_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = false;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    if (n_threads > threadpool->n_threads_max) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        htp_threadpool_free(threadpool);
    }

    return ret;
}
