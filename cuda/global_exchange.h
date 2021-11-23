#include "stream_manager.h"
#ifdef FMOE_USE_NCCL

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count, 
        long* global_expert_count, 
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->stream(0)));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->stream(0)));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    smgr->sync(1);
}

template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    /* TODO: may save for backward */
    long*expert_ptr = new long[n_expert * world_size];
    long*recv_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    long global_ptr = 0;
    //int global_idx = 0;
    for (size_t i = 0; i < n_expert; ++i) {
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            recv_ptr[idx] =  global_ptr;
            global_ptr += global_expert_count[idx];
            //global_idx++;
        }
    }

    int rank = 0;
    ncclCommUserRank(smgr->ncclcomm, &rank);
    int master_rank = (rank / 8) * 8;

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = master_rank; j < master_rank + 8; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[idx] * in_feat, 
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr[idx] * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            if (((j / 8) * 8) == master_rank) continue;
            int idx = i + j * n_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[idx] * in_feat, 
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr[idx] * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    delete [] recv_ptr;
}

template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[idx] * out_feat, 
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
}


#endif  // FMOE_USE_NCCL
