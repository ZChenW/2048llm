#include <cstdio>
#include <array>
#include <cuda/std/array>
#include <cuda/std/utility>
#include <curand_kernel.h>

typedef unsigned long long Board;

__device__ Board fill_rand(Board b, curandState * rng){
    int s = 0;
    for (int i = 0; i < 16; i ++){
        if (((b >> (i * 4)) & 0xF) == 0){
            s += 1;
        }
    }
    if (s == 0){
        return b;
    }
    int choice = curand_uniform(rng) * s;
    int val = curand_uniform(rng) < 0.1 ? 2 : 1;
    for (int i = 0; i < 16; i ++){
        if (((b >> (i * 4)) & 0xF) == 0){
            if (choice == 0){
                b |= ((Board)val << (i * 4));
                return b;
            }
            choice -= 1;
        }
    }
    return b;
}
__device__ Board transpose_board(Board b){
    Board b0 = b & 0xF0F0'0F0F'F0F0'0F0FuLL;
    Board b1 = b & 0x0000'F0F0'0000'F0F0uLL;
    Board b2 = b & 0x0F0F'0000'0F0F'0000uLL;
    b = b0 | (b1 << 12) | (b2 >> 12);
    b0 = b & 0xFF00'FF00'00FF'00FFuLL;
    b1 = b & 0x0000'0000'FF00'FF00uLL;
    b2 = b & 0x00FF'00FF'0000'0000uLL;
    b = b0 | (b1 << 24) | (b2 >> 24);
    return b;
}

__device__ Board hflip_board(Board b){
    Board b1 = b & 0x0F0F'0F0F'0F0F'0F0FuLL;
    b = ((b - b1) >> 4) | (b1 << 4);
    Board b2 = b & 0x00FF'00FF'00FF'00FFuLL;
    b = ((b - b2) >> 8) | (b2 << 8);
    return b;
}

__device__ cuda::std::pair<int, int> move_left(int key){
    int line[4];
    for (int i = 0; i < 4; i ++){
        line[i] = (key >> (i * 4)) & 0xF;
    }
    int new_line[4] = {0, 0, 0, 0};
    int idx = 0;
    bool merged = false;
    int score = 0;
    for (int i = 0; i < 4; i ++){
        if (line[i] == 0) continue;
        if (!merged && idx > 0 && new_line[idx - 1] == line[i] && line[i] != 15){
            new_line[idx - 1] += 1;
            score += (1 << new_line[idx - 1]);
            merged = true;
        }else{
            new_line[idx] = line[i];
            idx += 1;
            merged = false;
        }
    }
    int value = new_line[0] | (new_line[1] << 4) | (new_line[2] << 8) | (new_line[3] << 12);
    return {value, score};
}

__device__ cuda::std::array<cuda::std::pair<Board, int>, 4> next_moves(Board b){
    cuda::std::array<cuda::std::pair<Board, int>, 4> ret;
    // Left & Right
    Board b1 = hflip_board(b);
    Board new_b_left = 0, new_b_right = 0;
    int score_b_left = 0, score_b_right = 0;
    for (int i = 0; i < 4; i ++){
        auto [v_left, score_left] = move_left(((b >> (i * 16)) & 0xFFFF));
        new_b_left |= ((Board)v_left << (i * 16));
        score_b_left += score_left;
        auto [v_right, score_right] = move_left(((b1 >> (i * 16)) & 0xFFFF));
        new_b_right |= ((Board)v_right << (i * 16));
        score_b_right += score_right;
    }
    ret[0] = {new_b_left, score_b_left};
    ret[1] = {hflip_board(new_b_right), score_b_right};
    // Up & Down
    Board tb = transpose_board(b);
    Board b2 = hflip_board(tb);
    Board new_b_up = 0, new_b_down = 0;
    int score_b_up = 0, score_b_down = 0;
    for (int i = 0; i < 4; i ++){
        auto [v_up, score_up] = move_left(((tb >> (i * 16)) & 0xFFFF)); 
        new_b_up |= ((Board)v_up << (i * 16));
        score_b_up += score_up;
        auto [v_down, score_down] = move_left(((b2 >> (i * 16)) & 0xFFFF));
        new_b_down |= ((Board)v_down << (i * 16));
        score_b_down += score_down;
    }
    ret[2] = {transpose_board(new_b_up), score_b_up};
    ret[3] = {transpose_board(hflip_board(new_b_down)), score_b_down};
    for (int i = 0; i < 4; i ++){
        if (ret[i].first == b){
            ret[i] = {0, 0};
        }
    }
    return ret;
}
__device__ cuda::std::array<Board, 8> sym_copies(Board b){
    cuda::std::array<Board, 8> ret;
    Board tb = transpose_board(b);
    ret[0] = b;
    ret[1] = tb;
    ret[2] = hflip_board(ret[0]);
    ret[3] = hflip_board(ret[1]);
    ret[4] = transpose_board(ret[2]);
    ret[5] = transpose_board(ret[3]);
    ret[6] = hflip_board(ret[4]);
    ret[7] = hflip_board(ret[5]);
    return ret;
}

const int N_TUPLES = 8;
const int TUPLE_ELEMS = 6;

__constant__ int tuple_index_tables[N_TUPLES][TUPLE_ELEMS] = {
    {0, 1, 2, 4, 5, 6},
    {1, 2, 5, 6, 9, 13},
    {0, 1, 2, 3, 4, 5},
    {0, 1, 5, 6, 7, 10},
    {0, 1, 2, 5, 9, 10},
    {0, 1, 5, 9, 13, 14},
    {0, 1, 5, 8, 9, 13},
    {0, 1, 2, 4, 6, 10}
};

float network_weights[N_TUPLES][1 << (4 * TUPLE_ELEMS)];

typedef cuda::std::array<cuda::std::array<int, 8>, N_TUPLES> FeatureIndices;

__device__ FeatureIndices feature_indices(Board b){
    FeatureIndices ret;
    auto syms = sym_copies(b);
    for (int t = 0; t < N_TUPLES; t ++){
        for (int s = 0; s < 8; s ++){
            int idx = 0;
            for (int e = 0; e < TUPLE_ELEMS; e ++){
                int pos = tuple_index_tables[t][e];
                int v = (syms[s] >> (pos * 4)) & 0xF;
                idx |= v << (e * 4);
            }
            ret[t][s] = idx;
        }
    }
    return ret;
}
__device__ float get_network_score(const FeatureIndices & feat_idxs, float * network_weights){
    float total = 0;
    for (int t = 0; t < N_TUPLES; t ++){
        for (int s = 0; s < 8; s ++){
            int idx = feat_idxs[t][s];
            total += network_weights[(t << (4 * TUPLE_ELEMS)) + idx];
        }
    }
    return total / (8 * N_TUPLES);
}
__device__ void train_update_state(const FeatureIndices & feat_idxs, float delta, float learning_rate, float * network_weights){
    delta = delta / (8 * N_TUPLES) * learning_rate;
    for (int t = 0; t < N_TUPLES; t ++){
        for (int s = 0; s < 8; s ++){
            int idx = feat_idxs[t][s];
            network_weights[(t << (4 * TUPLE_ELEMS)) + idx] += delta;
        }
    }
}

__device__ cuda::std::pair<Board, int> train_episode(cuda::std::pair<Board, int> local_state, curandState * rng, float learning_rate, float * network_weights, int * step_total){
    auto [b, steps] = local_state;
    Board old_b = b;
    b = fill_rand(b, rng);
    steps += 1;
    auto nexts = next_moves(b);
    float best_score = 0;
    Board best_b = 0;
    for (int i = 0; i < 4; i ++){
        if (nexts[i].first != 0){
            float score = get_network_score(feature_indices(nexts[i].first), network_weights) + nexts[i].second;
            if (best_b == 0 || score > best_score){
                best_score = score;
                best_b = nexts[i].first;
            }
        }
    }
    // old_b -> best_b
    FeatureIndices old_feat_idxs = feature_indices(old_b);
    float old_score = get_network_score(old_feat_idxs, network_weights);
    train_update_state(old_feat_idxs, best_score - old_score, learning_rate, network_weights);
    if (best_b != 0){
        b = best_b;
    }else{
        atomicAdd(step_total + 0, 1);
        atomicAdd(step_total + 1, steps);
        b = 0;
        steps = 0;
    }
    return {b, steps};
}

__global__ void train_batch(cuda::std::pair<Board, int> * iter_states, curandState * rng_states, float learning_rate, float * network_weights, int step_cnt, int * step_total){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng = rng_states[idx];
    cuda::std::pair<Board, int> local_state = iter_states[idx];

    for (int i = 0; i < step_cnt; i ++){
        local_state = train_episode(local_state, &local_rng, learning_rate, network_weights, step_total);
    }
    rng_states[idx] = local_rng;
    iter_states[idx] = local_state;
}

__global__ void init_curand_states(curandState * states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, states + idx);
}

int main(){
    const float initV = 320000;
    for (int i = 0; i < N_TUPLES; i ++){
        for (int j = 0; j < (1 << (4 * TUPLE_ELEMS)); j ++){
            network_weights[i][j] = initV;
        }
    }

    float * network_weights_g;
    cudaMalloc(&network_weights_g, sizeof(network_weights));
    cudaMemcpy(network_weights_g, network_weights, sizeof(network_weights), cudaMemcpyHostToDevice);

    const int grid_size = 512;
    const int block_size = 32;
    const int nthreads = grid_size * block_size;
    curandState *dStates;
    cudaMalloc(&dStates, nthreads * sizeof(curandState));
    init_curand_states<<<grid_size, block_size>>>(dStates, 100);

    cuda::std::pair<Board, int> * iter_state_g;
    cudaMalloc(&iter_state_g, nthreads * sizeof(cuda::std::pair<Board, int>));
    cudaMemset(iter_state_g, 0, nthreads * sizeof(cuda::std::pair<Board, int>));

    int * step_total_g;
    cudaMalloc(&step_total_g, 2 * sizeof(int));

    int total_episodes = 1000'000'000;
    int cur_episodes =              0;

    int last_report_episodes = cur_episodes;
    int last_save_episodes = cur_episodes;
    int last_dump_episodes = cur_episodes;

    FILE * flog = fopen("train_log.txt", "a");

    int report_total_steps = 0;
    int report_total_episodes = 0;

    while (cur_episodes < total_episodes){
        cudaMemset(step_total_g, 0, 2 * sizeof(int));
        float lr = 0.1f;
        if (cur_episodes < 500'000'000){
            lr = 0.1f;
        }else{
            lr = 0.025f;
        }
        train_batch<<<grid_size, block_size>>>(iter_state_g, dStates, lr, network_weights_g, 1024, step_total_g);
        int step_total[2];    
        cudaMemcpy(step_total, step_total_g, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cur_episodes += step_total[0];

        report_total_episodes += step_total[0];
        report_total_steps += step_total[1];
        
        if (cur_episodes - last_report_episodes >= 10000){
            last_report_episodes += 10000;
            printf("i = %d steps_avg = %.2f\n", cur_episodes, (float)report_total_steps / report_total_episodes);
            fprintf(flog, "i = %d steps_avg = %.2f\n", cur_episodes, (float)report_total_steps / report_total_episodes);
            fflush(flog);
            report_total_episodes = 0;
            report_total_steps = 0;
        }
        if (cur_episodes - last_save_episodes >= 1000000){
            last_save_episodes += 1000000;
            cudaMemcpy(network_weights, network_weights_g, sizeof(network_weights), cudaMemcpyDeviceToHost);
            char filename[64];
            sprintf(filename, "./weights_latest.bin");
            FILE * fnet = fopen(filename, "wb");
            fwrite(network_weights, sizeof(network_weights), 1, fnet);
            fclose(fnet);
            if (cur_episodes - last_dump_episodes >= 5000000){
                last_dump_episodes += 5000000;
                sprintf(filename, "./weights_%d.bin", last_dump_episodes - 1);
                fnet = fopen(filename, "wb");
                fwrite(network_weights, sizeof(network_weights), 1, fnet);
                fclose(fnet);
            }
        }
    }

    return 0;
}
