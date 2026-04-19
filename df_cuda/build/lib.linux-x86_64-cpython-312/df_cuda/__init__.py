import torch as th
import numpy as np
from tqdm import tqdm
from . import _C
import time

@th.no_grad()
def fill_bits(X, X_processed, bV):
    _C.fill_bits(X, X_processed, bV);

@th.no_grad()
def count_same_tokens(X_processed, X1_processed, num_same_tokens, D, bV):
    _C.count_same_tokens(X_processed, X1_processed, num_same_tokens, D, bV);

@th.no_grad()
def calculate_val_tn1(num_same_tokens, max_nums, normalized_vals, gamma):
    _C.calculate_val_tn1(num_same_tokens, max_nums, normalized_vals, gamma);

@th.no_grad()
def calculate_val_te1(num_same_tokens, max_nums, normalized_vals):
    _C.calculate_val_te1(num_same_tokens, max_nums, normalized_vals);

@th.no_grad()
def normalize(normalized_vals, sum_vals):
    _C.normalize(normalized_vals, sum_vals);

@th.no_grad()
def sum_same_tokens(X_processed, X1_processed, normalized_vals, val_same_tokens, D, bV):
    _C.sum_same_tokens(X_processed, X1_processed, normalized_vals, val_same_tokens, D, bV);

@th.no_grad()
def sum_same_tokens_large(X_processed, X1_processed, normalized_vals, val_same_tokens, D, bV):
    _C.sum_same_tokens_large(X_processed, X1_processed, normalized_vals, val_same_tokens, D, bV);

@th.no_grad()
def get_thres(val_same_tokens, V, tau, kt, h):
    _C.get_thres(val_same_tokens, V, tau, kt, h);


# @th.no_grad()
# def inversion(X, X1, num_steps = 20, scheduler = "linear", tau = 1.0, debug = False):
#     if debug: start_time = time.time();

#     V = th.max(X1).item() + 1;
#     X1 = X1.to(th.uint64).contiguous();

#     B, N, D = X.shape[0], X1.shape[0], X.shape[1];
#     bV = int(np.ceil(np.log2(V)));
#     _64_bV = 64 // bV;
#     bD = (D - 1) // _64_bV + 1;

#     X_processed  = th.zeros((B, bD), dtype = th.uint64, device = X.device).contiguous();
#     X1_processed = th.zeros((N, bD), dtype = th.uint64, device = X.device).contiguous();
#     _C.fill_bits(X1, X1_processed, bV);

#     num_same_tokens = th.empty((B, N), dtype = th.int32, device = X.device).contiguous();
#     normalized_vals = th.empty((B, N), dtype = th.float32, device = X.device).contiguous();
#     val_same_tokens = th.empty((B, D), dtype = th.float32, device = X.device).contiguous();

#     iters = tqdm(range(num_steps, 0, -1)) if debug else range(num_steps, 0, -1);
#     for T in iters:
#         if debug: time0 = time.time();
#         t = T / num_steps;

#         if scheduler == "linear":
#             kt = t;
#             dkt = 1.0;
#         else:
#             del X_processed, X1_processed, num_same_tokens, normalized_vals, val_same_tokens;
#             th.cuda.empty_cache();
#             raise NotImplementedError;

#         _C.fill_bits(X.to(th.uint64).contiguous(), X_processed, bV);
        
#         if debug:
#             X_processed[0, 0].item();
#             time1 = time.time();

#         _C.count_same_tokens(X_processed, X1_processed, num_same_tokens, D, bV);
        
#         if debug:
#             num_same_tokens[0, 0].item();
#             time2 = time.time();

#         max_nums = th.max(num_same_tokens, dim = 1)[0].contiguous();

#         if t < 1.0:
#             _C.calculate_val_tn1(num_same_tokens, max_nums, normalized_vals, 1.0 + kt * V / (1 - kt));
#         else:
#             _C.calculate_val_te1(num_same_tokens, max_nums, normalized_vals);
        
#         sum_vals = th.sum(normalized_vals, dim = 1).contiguous();
#         _C.normalize(normalized_vals, sum_vals);
        
#         if debug:
#             normalized_vals[0, 0].item();
#             time3 = time.time();

#         _C.sum_same_tokens(X_processed, X1_processed, normalized_vals, val_same_tokens, D, bV);

#         if debug:
#             val_same_tokens[0, 0].item();
#             time4 = time.time();
        
#         h = dkt / num_steps;
#         _C.get_thres(val_same_tokens, V, tau, kt, h);

#         rand0 = th.rand_like(val_same_tokens);
#         randi = th.randint_like(val_same_tokens, high = V - 1);

#         X = th.where(rand0 < val_same_tokens, X, th.where(randi < X, randi, randi + 1));
#         X[0, 0].item();
    
#         if debug:
#             time5 = time.time();
#             iters.set_description("Lap0 {:.2f}s, Lap1 {:.2f}s, Lap2 {:.2f}s, Lap3 {:.2f}s, Lap4 {:.2f}s".format(
#                 time1 - time0, time2 - time1, time3 - time2, time4 - time3, time5 - time4
#             ));

#     if debug: print("{} Samples, Took {:.3f}s".format(B, time.time() - start_time));

#     del X_processed, X1_processed, num_same_tokens, normalized_vals, val_same_tokens;
#     th.cuda.empty_cache();

#     return X;





@th.no_grad()
def inversion(X, X1, V = None, num_steps = 20, scheduler = "linear", tau = 1.0, debug = False):
    if debug: start_time = time.time();

    if V is None:
        V = th.max(X1).item() + 1;
    X1 = X1.to(th.uint64).contiguous();

    B, N, D = X.shape[0], X1.shape[0], X.shape[1];
    bV = int(np.ceil(np.log2(V)));
    _64_bV = 64 // bV;
    bD = (D - 1) // _64_bV + 1;

    X_processed  = th.zeros((B, bD), dtype = th.uint64, device = X.device).contiguous();
    X1_processed = th.zeros((N, bD), dtype = th.uint64, device = X.device).contiguous();
    _C.fill_bits(X1, X1_processed, bV);

    num_same_tokens = th.empty((B, N), dtype = th.int32, device = X.device).contiguous();
    normalized_vals = th.empty((B, N), dtype = th.float32, device = X.device).contiguous();
    val_same_tokens = th.empty((B, D), dtype = th.float32, device = X.device).contiguous();

    iters = tqdm(range(num_steps, 0, -1)) if debug else range(num_steps, 0, -1);
    for T in iters:
        if debug: time0 = time.time();
        t = T / num_steps;

        if scheduler == "linear":
            kt = t;
            dkt = 1.0;
        else:
            del X_processed, X1_processed, num_same_tokens, normalized_vals, val_same_tokens;
            th.cuda.empty_cache();
            raise NotImplementedError;

        _C.fill_bits(X.to(th.uint64).contiguous(), X_processed, bV);
        
        if debug:
            X_processed[0, 0].item();
            time1 = time.time();

        _C.count_same_tokens(X_processed, X1_processed, num_same_tokens, D, bV);
        
        if debug:
            num_same_tokens[0, 0].item();
            time2 = time.time();

        max_nums = th.max(num_same_tokens, dim = 1)[0].contiguous();

        if t < 1.0:
            _C.calculate_val_tn1(num_same_tokens, max_nums, normalized_vals, 1.0 + kt * V / (1 - kt));
        else:
            _C.calculate_val_te1(num_same_tokens, max_nums, normalized_vals);
        
        sum_vals = th.sum(normalized_vals, dim = 1).contiguous();
        _C.normalize(normalized_vals, sum_vals);
        
        if debug:
            normalized_vals[0, 0].item();
            time3 = time.time();
        
        val_same_tokens.zero_();

        _C.sum_same_tokens_large(X_processed, X1_processed, normalized_vals, val_same_tokens, D, bV);

        if debug:
            val_same_tokens[0, 0].item();
            time4 = time.time();
        
        h = dkt / num_steps;
        _C.get_thres(val_same_tokens, V, tau, kt, h);

        rand0 = th.rand_like(val_same_tokens);
        randi = th.randint_like(val_same_tokens, high = V - 1);

        X = th.where(rand0 < val_same_tokens, X, th.where(randi < X, randi, randi + 1));
        X[0, 0].item();
    
        if debug:
            time5 = time.time();
            iters.set_description("Lap0 {:.2f}s, Lap1 {:.2f}s, Lap2 {:.2f}s, Lap3 {:.2f}s, Lap4 {:.2f}s".format(
                time1 - time0, time2 - time1, time3 - time2, time4 - time3, time5 - time4
            ));

    if debug: print("{} Samples, Took {:.3f}s".format(B, time.time() - start_time));

    del X_processed, X1_processed, num_same_tokens, normalized_vals, val_same_tokens;
    th.cuda.empty_cache();

    return X;