#include "htp-ops.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static_assert(sizeof(htp_opbatch_rsp) == 24, "response wire ABI changed");
static_assert(offsetof(htp_opbatch_rsp, failed_op) == 20, "failed-op wire slot changed");

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    htp_status_counts counts = {};
    htp_status_counts_reset(&counts);

    htp_opbatch_rsp ok = {};
    ok.id              = 0;
    ok.status          = HTP_STATUS_OK;
    ok.n_bufs          = 2;
    ok.n_tensors       = 7;
    ok.n_ops           = 3;
    ok.failed_op       = HTP_OP_INDEX_NONE;

    require(htp_opbatch_rsp_is_consistent(&ok), "valid success response rejected");
    require(htp_opbatch_rsp_attempted(&ok) == 3, "success attempted count");
    require(htp_opbatch_rsp_completed(&ok) == 3, "success completed count");
    htp_status_counts_record_submission(&counts, ok.n_ops);
    htp_status_counts_record_response(&counts, &ok);

    htp_opbatch_rsp invalid_params = ok;
    invalid_params.id              = 1;
    invalid_params.status          = HTP_STATUS_INVAL_PARAMS;
    invalid_params.failed_op       = 1;

    require(htp_opbatch_rsp_is_consistent(&invalid_params), "valid failure response rejected");
    require(htp_opbatch_rsp_attempted(&invalid_params) == 2, "failure attempted count");
    require(htp_opbatch_rsp_completed(&invalid_params) == 1, "failure completed count");
    htp_status_counts_record_submission(&counts, invalid_params.n_ops);
    htp_status_counts_record_response(&counts, &invalid_params);

    htp_opbatch_rsp unsupported = ok;
    unsupported.id              = 2;
    unsupported.status          = HTP_STATUS_NO_SUPPORT;
    unsupported.n_ops           = 1;
    unsupported.failed_op       = 0;

    require(htp_opbatch_rsp_is_consistent(&unsupported), "valid unsupported response rejected");
    htp_status_counts_record_submission(&counts, unsupported.n_ops);
    htp_status_counts_record_response(&counts, &unsupported);

    require(counts.submitted == 7, "submitted total");
    require(counts.completed == 4, "completed total");
    require(counts.failed == 2, "failed total");
    require(counts.unsupported == 1, "unsupported total");
    require(counts.response_errors == 0, "response-error total");
    require(!htp_status_poison_session(HTP_STATUS_NO_SUPPORT, true), "NO_SUPPORT poisoned session");
    require(!htp_status_poison_session(HTP_STATUS_INVAL_PARAMS, true), "INVAL_PARAMS poisoned session");
    require(!htp_status_poison_session(HTP_STATUS_VTCM_TOO_SMALL, true), "VTCM failure poisoned session");
    require(htp_status_poison_session(HTP_STATUS_INTERNAL_ERR, true), "INTERNAL_ERR did not poison session");
    require(htp_status_poison_session(HTP_STATUS_OK, false), "malformed response did not poison session");

    htp_opbatch_rsp malformed_success = ok;
    malformed_success.failed_op       = 0;
    require(!htp_opbatch_rsp_is_consistent(&malformed_success), "success response with failed op accepted");

    htp_opbatch_rsp malformed_failure = invalid_params;
    malformed_failure.failed_op       = HTP_OP_INDEX_NONE;
    require(!htp_opbatch_rsp_is_consistent(&malformed_failure), "failure response without failed op accepted");

    malformed_failure.failed_op = malformed_failure.n_ops;
    require(!htp_opbatch_rsp_is_consistent(&malformed_failure), "out-of-range failed op accepted");

    htp_opbatch_rsp unknown_status = ok;
    unknown_status.status          = 0;
    require(!htp_opbatch_rsp_is_consistent(&unknown_status), "unknown status accepted");

    htp_opbatch_rsp empty_success = ok;
    empty_success.n_ops           = 0;
    require(!htp_opbatch_rsp_is_consistent(&empty_success), "empty response accepted");

    std::puts("HTP opbatch status tests passed");
    return 0;
}
