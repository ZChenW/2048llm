#include <cstdio>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
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

float network_weights_value[N_TUPLES][1 << (4 * TUPLE_ELEMS)];

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

__device__ cuda::std::pair<Board, int> train_episode(cuda::std::pair<Board, int> local_state, curandState * rng, float learning_rate, float * network_weights_value, int * step_total){
    auto [b, steps] = local_state;

    int max_tile = 0;
    for (int i = 0; i < 16; i ++){
        int v = (b >> (i * 4)) & 0xF;
        if (v > max_tile){
            max_tile = v;
        }
    }
    Board old_b = b;
    float best_score = 0;
    Board best_b = 0;
    if (max_tile >= 13){
        best_b = 0;
        best_score = 1 << max_tile;
    }else{
        b = fill_rand(b, rng);
        steps += 1;
        auto nexts = next_moves(b);
        for (int i = 0; i < 4; i ++){
            if (nexts[i].first != 0){
                float score = get_network_score(feature_indices(nexts[i].first), network_weights_value);
                if (best_b == 0 || score > best_score){
                    best_score = score;
                    best_b = nexts[i].first;
                }
            }
        }
        if (best_b == 0){
            best_score = 1 << max_tile;
        }
    }
    // old_b -> best_b
    FeatureIndices old_feat_idxs = feature_indices(old_b);
    float old_score = get_network_score(old_feat_idxs, network_weights_value);
    train_update_state(old_feat_idxs, best_score - old_score, learning_rate, network_weights_value);
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

__global__ void train_batch(cuda::std::pair<Board, int> * iter_states, curandState * rng_states, float learning_rate, float * network_weights_value, int step_cnt, int * step_total){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_rng = rng_states[idx];
    cuda::std::pair<Board, int> local_state = iter_states[idx];

    for (int i = 0; i < step_cnt; i ++){
        local_state = train_episode(local_state, &local_rng, learning_rate, network_weights_value, step_total);
    }
    rng_states[idx] = local_rng;
    iter_states[idx] = local_state;
}

__global__ void init_curand_states(curandState * states, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, states + idx);
}

struct TrainConfig {
    long long episodes = 200000000LL;
    float lr = 0.1f;
    unsigned long seed = 100;
    std::string out_dir = ".";
    long long save_every = 1000000LL;
    std::string resume_path;
};

void print_usage(const char * argv0){
    printf("Usage: %s [options]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --help              Show this help message\n");
    printf("  --episodes N        Episodes to train in this invocation (default: 200000000)\n");
    printf("  --lr FLOAT          Learning rate (default: 0.1)\n");
    printf("  --seed INT          CUDA RNG seed (default: 100)\n");
    printf("  --out-dir PATH      Output directory (default: .)\n");
    printf("  --save-every N      Checkpoint interval in episodes (default: 1000000)\n");
    printf("  --resume PATH       Load raw float32 weights before training\n");
}

bool parse_long_long(const char * text, long long * out){
    char * end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno || end == text || *end != '\0' || value < 0){
        return false;
    }
    *out = value;
    return true;
}

bool parse_ulong(const char * text, unsigned long * out){
    char * end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno || end == text || *end != '\0'){
        return false;
    }
    *out = value;
    return true;
}

bool parse_float(const char * text, float * out){
    char * end = nullptr;
    errno = 0;
    float value = std::strtof(text, &end);
    if (errno || end == text || *end != '\0'){
        return false;
    }
    *out = value;
    return true;
}

bool parse_args(int argc, char ** argv, TrainConfig * config){
    for (int i = 1; i < argc; i ++){
        const char * arg = argv[i];
        if (std::strcmp(arg, "--help") == 0){
            print_usage(argv[0]);
            std::exit(0);
        }else if (std::strcmp(arg, "--episodes") == 0 && i + 1 < argc){
            if (!parse_long_long(argv[++i], &config->episodes)) return false;
        }else if (std::strcmp(arg, "--lr") == 0 && i + 1 < argc){
            if (!parse_float(argv[++i], &config->lr)) return false;
        }else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc){
            if (!parse_ulong(argv[++i], &config->seed)) return false;
        }else if (std::strcmp(arg, "--out-dir") == 0 && i + 1 < argc){
            config->out_dir = argv[++i];
        }else if (std::strcmp(arg, "--save-every") == 0 && i + 1 < argc){
            if (!parse_long_long(argv[++i], &config->save_every)) return false;
        }else if (std::strcmp(arg, "--resume") == 0 && i + 1 < argc){
            config->resume_path = argv[++i];
        }else{
            return false;
        }
    }
    return config->episodes >= 0 && config->save_every > 0;
}

bool ensure_dir(const std::string& path){
    if (path.empty()) return false;

    std::string current;
    for (size_t i = 0; i < path.size(); i++){
        current.push_back(path[i]);

        if (path[i] == '/' || i + 1 == path.size()){
            std::string dir = current;
            while (!dir.empty() && dir.back() == '/'){
                dir.pop_back();
            }
            if (dir.empty() || dir == ".") continue;
            if (mkdir(dir.c_str(), 0775) != 0 && errno != EEXIST){
                perror(dir.c_str());
                return false;
            }
        }
    }
    return true;
}

std::string join_path(const std::string& dir, const std::string& name){
    if (dir.empty() || dir == ".") return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

bool write_weights_file(const std::string& path){
    FILE * fnet = fopen(path.c_str(), "wb");
    if (!fnet){
        fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        return false;
    }
    size_t wrote = fwrite(network_weights_value, sizeof(network_weights_value), 1, fnet);
    fclose(fnet);
    if (wrote != 1){
        fprintf(stderr, "failed to write complete weights to %s\n", path.c_str());
        return false;
    }
    return true;
}

bool load_weights_file(const std::string& path){
    FILE * fnet = fopen(path.c_str(), "rb");
    if (!fnet){
        fprintf(stderr, "failed to open resume weights %s\n", path.c_str());
        return false;
    }
    size_t read = fread(network_weights_value, sizeof(network_weights_value), 1, fnet);
    int extra = fgetc(fnet);
    fclose(fnet);
    if (read != 1 || extra != EOF){
        fprintf(stderr, "resume weights must be exactly %zu bytes: %s\n", sizeof(network_weights_value), path.c_str());
        return false;
    }
    return true;
}

bool write_metadata_file(const std::string& path, long long episodes, const TrainConfig& config){
    FILE * f = fopen(path.c_str(), "w");
    if (!f){
        fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        return false;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"episodes\": %lld,\n", episodes);
    fprintf(f, "  \"lr\": %.9g,\n", config.lr);
    fprintf(f, "  \"seed\": %lu,\n", config.seed);
    fprintf(f, "  \"mode\": \"highwin8192\",\n");
    fprintf(f, "  \"tuple_count\": %d,\n", N_TUPLES);
    fprintf(f, "  \"tuple_len\": %d,\n", TUPLE_ELEMS);
    fprintf(f, "  \"alphabet_size\": 16,\n");
    fprintf(f, "  \"weight_file_size\": %zu\n", sizeof(network_weights_value));
    fprintf(f, "}\n");
    fclose(f);
    return true;
}

bool save_checkpoint(const TrainConfig& config, long long episodes){
    if (!ensure_dir(config.out_dir)){
        fprintf(stderr, "failed to create output directory %s\n", config.out_dir.c_str());
        return false;
    }

    std::string latest_bin = join_path(config.out_dir, "weights_latest.bin");
    std::string latest_json = join_path(config.out_dir, "weights_latest.json");
    char ckpt_name[128];
    char json_name[128];
    snprintf(ckpt_name, sizeof(ckpt_name), "ckpt_%lld.bin", episodes);
    snprintf(json_name, sizeof(json_name), "ckpt_%lld.json", episodes);

    if (!write_weights_file(latest_bin)) return false;
    if (!write_metadata_file(latest_json, episodes, config)) return false;
    if (!write_weights_file(join_path(config.out_dir, ckpt_name))) return false;
    if (!write_metadata_file(join_path(config.out_dir, json_name), episodes, config)) return false;
    printf("saved checkpoint at %lld episodes to %s\n", episodes, config.out_dir.c_str());
    return true;
}

int main(int argc, char ** argv){
    TrainConfig config;
    if (!parse_args(argc, argv, &config)){
        fprintf(stderr, "invalid arguments\n");
        print_usage(argv[0]);
        return 1;
    }

    for (int t = 0; t < N_TUPLES; t ++){
        for (int i = 0; i < (1 << (4 * TUPLE_ELEMS)); i ++){
            network_weights_value[t][i] = 8192.0f;
        }
    }
    if (!config.resume_path.empty() && !load_weights_file(config.resume_path)){
        return 1;
    }
    if (!ensure_dir(config.out_dir)){
        fprintf(stderr, "failed to create output directory %s\n", config.out_dir.c_str());
        return 1;
    }

    float * network_weights_value_g;
    cudaMalloc(&network_weights_value_g, sizeof(network_weights_value));
    cudaMemcpy(network_weights_value_g, network_weights_value, sizeof(network_weights_value), cudaMemcpyHostToDevice);

    const int grid_size = 512;
    const int block_size = 32;
    const int nthreads = grid_size * block_size;
    curandState *dStates;
    cudaMalloc(&dStates, nthreads * sizeof(curandState));
    init_curand_states<<<grid_size, block_size>>>(dStates, config.seed);

    cuda::std::pair<Board, int> * iter_state_g;
    cudaMalloc(&iter_state_g, nthreads * sizeof(cuda::std::pair<Board, int>));
    cudaMemset(iter_state_g, 0, nthreads * sizeof(cuda::std::pair<Board, int>));

    int * step_total_g;
    cudaMalloc(&step_total_g, 2 * sizeof(int));

    long long total_episodes = config.episodes;
    long long cur_episodes = 0;

    long long last_report_episodes = cur_episodes;
    long long next_save_episodes = config.save_every;
    long long last_saved_episodes = -1;

    FILE * flog = fopen(join_path(config.out_dir, "train_log_2048.txt").c_str(), "a");
    if (!flog){
        fprintf(stderr, "failed to open training log\n");
        return 1;
    }

    int report_total_steps = 0;
    int report_total_episodes = 0;

    while (cur_episodes < total_episodes){
        cudaMemset(step_total_g, 0, 2 * sizeof(int));
        float lr = config.lr;

        train_batch<<<grid_size, block_size>>>(iter_state_g, dStates, lr, network_weights_value_g, 1024, step_total_g);
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
        while (cur_episodes >= next_save_episodes && next_save_episodes <= total_episodes){
            cudaMemcpy(network_weights_value, network_weights_value_g, sizeof(network_weights_value), cudaMemcpyDeviceToHost);
            if (!save_checkpoint(config, next_save_episodes)){
                fclose(flog);
                return 1;
            }
            last_saved_episodes = next_save_episodes;
            next_save_episodes += config.save_every;
        }
    }

    cudaMemcpy(network_weights_value, network_weights_value_g, sizeof(network_weights_value), cudaMemcpyDeviceToHost);
    if (last_saved_episodes != total_episodes && !save_checkpoint(config, total_episodes)){
        fclose(flog);
        return 1;
    }
    fclose(flog);

    return 0;
}
