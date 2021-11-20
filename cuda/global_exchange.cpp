#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>
#include <cuda_fp16.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>

extern "C" size_t array_min_max_size_f16_host(
    at::Half *input, 
    int input_num_element,
    at::Half *output,
    cudaStream_t stream);

extern "C" void compress_f16_to_uint8_host_vector(
    at::Half *input, 
    int input_num_element,
    int max_chunk_size,
    int num_chunks,
    long* chunks_offset,
    uint8_t *output,
    long* outputs_offset,
    at::Half* min_max,
    //void *dev_buffer,
    //size_t dev_size,
    cudaStream_t stream);

extern "C" void decompress_uint8_to_f16_host_vector(
    uint8_t *input,
    int max_chunk_size,
    int num_chunks,
    long* inputs_offset,
    at::Half *output,
    long* outputs_offset,
    cudaStream_t stream);

int align_size(int size, int align) {
  return ((size) + (align) - 1) / (align) * (align);
}

int get_compressed_buffer_size(int n_chunks, int chunk_size) {
  int align_bytes = 32;
  int compressed_align_bytes = align_size(chunk_size, align_bytes) * n_chunks;
  int min_max_align_bytes = align_size(2 * 2, align_bytes) * n_chunks;
  return compressed_align_bytes + min_max_align_bytes;
}


torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return global_expert_count;
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    int max_chunk_size = local_expert_count.max().data_ptr<long>()[0] * in_feat;
    int global_max_chunk_size = global_expert_count.max().data_ptr<long>()[0] * in_feat;
    int num_chunks = local_expert_count.size(0);

    // local
    auto local_offset = local_expert_count.cumsum(0) * in_feat;
    auto local_compressed_offset = local_offset.clone();
    long* local_expert_count_ptr = local_expert_count.data_ptr<long>();
    long* local_expert_compress_ptr = new long[num_chunks];
    long* local_compressed_offset_ptr = local_compressed_offset.data_ptr<long>();
    long local_com_offset = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        local_expert_compress_ptr[i] = local_expert_count_ptr[i] * in_feat;
        if (local_expert_count_ptr[i]) {
          local_expert_compress_ptr[i] += 32;
          local_com_offset += 32;
        }
        local_compressed_offset_ptr[i] += local_com_offset;
    }
    auto local_compressed = input_buf.new_empty({local_compressed_offset_ptr[num_chunks - 1]}, at::ScalarType::Byte);

    // temp_buff min_max
    auto min_max = input_buf.new_empty({2}, at::ScalarType::Half);
    min_max[0] = input_buf.min();
    min_max[1] = input_buf.max();
    //size_t temp_buff_size = array_min_max_size_f16_host(
    //    input_buf.data_ptr<at::Half>(),
    //    input_buf.numel(),
    //    reinterpret_cast<at::Half*>(local_compressed.data_ptr<uint8_t>()),
    //    smgr->stream(0));
    //auto temp_buff = input_buf.new_empty({temp_buff_size}, at::ScalarType::Byte);

    // compress
    compress_f16_to_uint8_host_vector(
        input_buf.data_ptr<at::Half>(),
        input_buf.numel(),
        max_chunk_size,
        num_chunks,
        local_offset.cuda().data_ptr<long>(),
        local_compressed.data_ptr<uint8_t>(),
        local_compressed_offset.cuda().data_ptr<long>(),
        min_max.data_ptr<at::Half>(),
        //temp_buff.data_ptr<uint8_t>(),
        //temp_buff_size,
        smgr->stream(0));

    // global
    auto global_offset = global_expert_count.new_empty({num_chunks});
    auto global_compressed_offset = global_offset.clone();
    long* global_expert_count_ptr = global_expert_count.data_ptr<long>();
    long* global_expert_compress_ptr = new long[num_chunks];
    long* global_offset_ptr = global_offset.data_ptr<long>();
    long* global_compressed_offset_ptr = global_compressed_offset.data_ptr<long>();
    long pre_global_offset = 0;
    long pre_global_com_offset = 0;
    int global_idx = 0;
    for (size_t i = 0; i < n_expert; ++i) {
      for (size_t j = 0; j < n_workers; ++j) {
        int idx = i + j * n_expert;
        global_expert_compress_ptr[idx] = global_expert_count_ptr[idx] * in_feat;
        if (global_expert_count_ptr[idx]) {
          global_expert_compress_ptr[idx] += 32;
        }

        global_offset_ptr[global_idx] = global_expert_count_ptr[idx] * in_feat + pre_global_offset;
        global_compressed_offset_ptr[global_idx] = global_expert_compress_ptr[idx] + pre_global_com_offset;

        pre_global_offset = global_offset_ptr[global_idx];
        pre_global_com_offset = global_compressed_offset_ptr[global_idx];
        global_idx++;
      }
    }

    auto global_compressed = input_buf.new_empty({global_compressed_offset_ptr[num_chunks - 1]}, at::ScalarType::Byte);

    // alltoall
    AT_DISPATCH_INTEGRAL_TYPES(local_compressed.scalar_type(),
            "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            local_compressed.data_ptr<scalar_t>(),
            local_expert_compress_ptr,
            global_expert_compress_ptr,
            global_compressed.data_ptr<scalar_t>(),
            1, n_expert, n_workers,
            smgr
        );
    }));

    // decompress
    decompress_uint8_to_f16_host_vector(
        global_compressed.data_ptr<uint8_t>(),
        global_max_chunk_size,
        num_chunks,
        global_compressed_offset.cuda().data_ptr<long>(),
        global_input_buf.data_ptr<at::Half>(),
        global_offset.cuda().data_ptr<long>(),
        smgr->stream(0));

    delete[] local_expert_compress_ptr;
    delete[] global_expert_compress_ptr;
    smgr->sync(1);
    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    int max_chunk_size = local_expert_count.max().data_ptr<long>()[0] * out_feat;
    int global_max_chunk_size = global_expert_count.max().data_ptr<long>()[0] * out_feat;
    int num_chunks = local_expert_count.size(0);

    // global
    auto global_offset = global_expert_count.new_empty({num_chunks});
    auto global_compressed_offset = global_offset.clone();
    long* global_expert_count_ptr = global_expert_count.data_ptr<long>();
    long* global_expert_compress_ptr = new long[num_chunks];
    long* global_offset_ptr = global_offset.data_ptr<long>();
    long* global_compressed_offset_ptr = global_compressed_offset.data_ptr<long>();
    long pre_global_offset = 0;
    long pre_global_com_offset = 0;
    int global_idx = 0;
    for (size_t i = 0; i < n_expert; ++i) {
      for (size_t j = 0; j < n_workers; ++j) {
        int idx = i + j * n_expert;
        global_expert_compress_ptr[idx] = global_expert_count_ptr[idx] * out_feat;
        if (global_expert_count_ptr[idx]) {
          global_expert_compress_ptr[idx] += 32;
        }

        global_offset_ptr[global_idx] = global_expert_count_ptr[idx] * out_feat + pre_global_offset;
        global_compressed_offset_ptr[global_idx] = global_expert_compress_ptr[idx] + pre_global_com_offset;

        pre_global_offset = global_offset_ptr[global_idx];
        pre_global_com_offset = global_compressed_offset_ptr[global_idx];
        global_idx++;
      }
    }
    auto global_compressed = output_buf.new_empty({global_compressed_offset_ptr[num_chunks - 1]}, at::ScalarType::Byte);

    // temp_buff min_max
    auto min_max = output_buf.new_empty({2}, at::ScalarType::Half);
    min_max[0] = output_buf.min();
    min_max[1] = output_buf.max();
    //size_t temp_buff_size = array_min_max_size_f16_host(
    //    output_buf.data_ptr<at::Half>(),
    //    output_buf.numel(),
    //    reinterpret_cast<at::Half*>(global_compressed.data_ptr<uint8_t>()),
    //    smgr->stream(0));
    //auto temp_buff = output_buf.new_empty({temp_buff_size}, at::ScalarType::Byte);

    // compress
    compress_f16_to_uint8_host_vector(
        output_buf.data_ptr<at::Half>(),
        output_buf.numel(),
        global_max_chunk_size,
        num_chunks,
        global_offset.cuda().data_ptr<long>(),
        global_compressed.data_ptr<uint8_t>(),
        global_compressed_offset.cuda().data_ptr<long>(),
        min_max.data_ptr<at::Half>(),
        //temp_buff.data_ptr<uint8_t>(),
        //temp_buff_size,
        smgr->stream(0));

    // local
    auto local_offset = local_expert_count.cumsum(0) * out_feat;
    auto local_compressed_offset = local_offset.clone();
    long* local_expert_count_ptr = local_expert_count.data_ptr<long>();
    long* local_expert_compress_ptr = new long[num_chunks];
    long* local_compressed_offset_ptr = local_compressed_offset.data_ptr<long>();
    long local_com_offset = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        local_expert_compress_ptr[i] = local_expert_count_ptr[i] * out_feat;
        if (local_expert_count_ptr[i]) {
          local_expert_compress_ptr[i] += 32;
          local_com_offset += 32;
        }
        local_compressed_offset_ptr[i] += local_com_offset;
    }
    auto local_compressed = output_buf.new_empty({local_compressed_offset_ptr[num_chunks - 1]}, at::ScalarType::Byte);

    // alltoall
    AT_DISPATCH_INTEGRAL_TYPES(global_compressed.scalar_type(),
            "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            global_compressed.data_ptr<scalar_t>(),
            local_expert_compress_ptr,
            global_expert_compress_ptr,
            local_compressed.data_ptr<scalar_t>(),
            1, n_expert, n_workers,
            smgr
        );
    }));

    // decompress
    decompress_uint8_to_f16_host_vector(
        local_compressed.data_ptr<uint8_t>(),
        max_chunk_size,
        num_chunks,
        local_compressed_offset.cuda().data_ptr<long>(),
        local_output_buf.data_ptr<at::Half>(),
        local_offset.cuda().data_ptr<long>(),
        smgr->stream(0));

    delete[] local_expert_compress_ptr;
    delete[] global_expert_compress_ptr;
    smgr->sync(1);
    return local_output_buf;
}

#include <c10d/ProcessGroupNCCL.hpp>

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

#endif  // FMOE_USE_NCCL
