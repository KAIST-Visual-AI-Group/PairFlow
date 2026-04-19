#pragma once
#include <torch/extension.h>

void fill_bits(
    torch::Tensor& data,
    torch::Tensor& out,
    uint64_t num_bits
);

void count_same_tokens(
    torch::Tensor& x0,
    torch::Tensor& x1,
    torch::Tensor& out,
    uint64_t D,
    uint64_t num_bits
);

// void sum_same_tokens(
//     torch::Tensor& x0,
//     torch::Tensor& x1,
//     torch::Tensor& val,
//     torch::Tensor& out,
//     uint64_t D,
//     uint64_t num_bits
// );

void sum_same_tokens_large(
    torch::Tensor& x0,
    torch::Tensor& x1,
    torch::Tensor& val,
    torch::Tensor& out,
    uint64_t D,
    uint64_t num_bits
);

void calculate_val_tn1(
    torch::Tensor& num,
    torch::Tensor& max_nums,
    torch::Tensor& out,
    float gamma
);

void calculate_val_te1(
    torch::Tensor& num,
    torch::Tensor& max_nums,
    torch::Tensor& out
);

void normalize(
    torch::Tensor& val,
    torch::Tensor& sum_vals
);


void get_thres(
    torch::Tensor& val,
    int V,
    float tau,
    float t,
    float h
);
