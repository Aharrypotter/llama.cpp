#include "htp-cpu-impl.h"

// Barrier implementation
void ggml_barrier_htp(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
}

// NUMA affinity helper
#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

// ARM architecture features initialization
#if defined(__ARM_ARCH)

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1 << 13)
#endif

void ggml_init_arm_arch_features(void) {
#if defined(__linux__) && defined(__aarch64__)
    uint32_t hwcap = getauxval(AT_HWCAP);
    uint32_t hwcap2 = getauxval(AT_HWCAP2);

    ggml_arm_arch_features.has_neon = !!(hwcap & HWCAP_ASIMD);
    ggml_arm_arch_features.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    ggml_arm_arch_features.has_i8mm = !!(hwcap2 & HWCAP2_I8MM);
    ggml_arm_arch_features.has_sve  = !!(hwcap & HWCAP_SVE);

#if defined(__ARM_FEATURE_SVE)
    ggml_arm_arch_features.sve_cnt = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
#else
    ggml_arm_arch_features.sve_cnt = 0;
#endif

#elif defined(__APPLE__)
    ggml_arm_arch_features.has_neon = 1;
    ggml_arm_arch_features.has_dotprod = 1;
    ggml_arm_arch_features.has_i8mm = 1;
    ggml_arm_arch_features.has_sve  = 0;
    ggml_arm_arch_features.sve_cnt = 0;
#else
    ggml_arm_arch_features.has_neon = 1;
    ggml_arm_arch_features.has_dotprod = 0;
    ggml_arm_arch_features.has_i8mm = 0;
    ggml_arm_arch_features.has_sve  = 0;
    ggml_arm_arch_features.sve_cnt = 0;
#endif
}
#endif // __ARM_ARCH
