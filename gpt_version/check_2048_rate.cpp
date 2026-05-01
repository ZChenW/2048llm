#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <array>
#include <mutex>
#include <atomic>
#include <thread>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
typedef unsigned long long Board;

void display_board(Board b) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int v = (b >> (i * 4 * 4 + j * 4)) & 0xF;
            if (v == 0){
                printf("     |");
            }else{
                printf("%5d|", 1 << v);
            }
        }
        printf("\n");
    }
}

Board fill_rand(Board b, std::mt19937 &rng){
    int s = 0;
    for (int i = 0; i < 16; i ++){
        if (((b >> (i * 4)) & 0xF) == 0){
            s += 1;
        }
    }
    if (s == 0){
        return b;
    }
    std::uniform_int_distribution<int> dist(0, s - 1);
    std::uniform_int_distribution<int> dist_2(0, 9);
    int choice = dist(rng);
    int val = dist_2(rng) == 0 ? 2 : 1;
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
std::pair<int, int> move_lut_left[65536];

void fill_move_lut(){
    for (int a = 0; a < 16; a ++){
        for (int b = 0; b < 16; b ++){
            for (int c = 0; c < 16; c ++){
                for (int d = 0; d < 16; d ++){
                    int line[4] = {a, b, c, d};
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
                    int key = a | (b << 4) | (c << 8) | (d << 12);
                    int value = new_line[0] | (new_line[1] << 4) | (new_line[2] << 8) | (new_line[3] << 12);
                    move_lut_left[key] = {value, score};
                }
            }
        }
    }
}

Board transpose_board(Board b){
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

Board hflip_board(Board b){
    Board new_b = 0;
    for (int i = 0; i < 4; i ++){
        new_b |= ((b >> (i * 4)) & 0x000F'000F'000F'000FuLL) << ((3 - i) * 4);
    }
    return new_b;
}

std::array<std::pair<Board, int>, 4> next_moves(Board b){
    std::array<std::pair<Board, int>, 4> ret;
    // Left & Right
    Board b1 = hflip_board(b);
    Board new_b_left = 0, new_b_right = 0;
    int score_b_left = 0, score_b_right = 0;
    for (int i = 0; i < 4; i ++){
        auto [v_left, score_left] = move_lut_left[((b >> (i * 16)) & 0xFFFF)];
        new_b_left |= ((Board)v_left << (i * 16));
        score_b_left += score_left;
        auto [v_right, score_right] = move_lut_left[((b1 >> (i * 16)) & 0xFFFF)];
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
        auto [v_up, score_up] = move_lut_left[((tb >> (i * 16)) & 0xFFFF)];
        new_b_up |= ((Board)v_up << (i * 16));
        score_b_up += score_up;
        auto [v_down, score_down] = move_lut_left[((b2 >> (i * 16)) & 0xFFFF)];
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

std::array<Board, 8> sym_copies(Board b){
    std::array<Board, 8> ret;
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

const int tuple_index_tables[N_TUPLES][TUPLE_ELEMS] = {
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

typedef std::array<std::array<int, 8>, N_TUPLES> FeatureIndices;

FeatureIndices feature_indices(Board b){
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

float get_network_score(const FeatureIndices & feat_idxs){
    float total = 0;
    for (int t = 0; t < N_TUPLES; t ++){
        for (int s = 0; s < 8; s ++){
            int idx = feat_idxs[t][s];
            total += network_weights[t][idx];
        }
    }
    return total / (8 * N_TUPLES);
}

float get_network_score(Board b){
    return get_network_score(feature_indices(b));
}

float eval_kply(Board b, int k, std::unordered_map<Board, float> * cache){
    if (k == 0){
        return get_network_score(b);
    }
    if (cache[k].find(b) != cache[k].end()){
        return cache[k][b];
    }
    int cnt = 0;
    float avg_score = 0;
    for (int i = 0; i < 16; i ++){
        if (((b >> (i * 4)) & 0xF) == 0){
            cnt += 1;
            for (int val = 1; val <= 2; val ++){
                Board b2 = b | ((Board)val << (i * 4));
                auto nexts = next_moves(b2);
                float best_score = 0;
                for (int j = 0; j < 4; j ++){
                    if (nexts[j].first != 0){
                        float score = eval_kply(nexts[j].first, k - 1, cache);// + nexts[j].second;
                        if (best_score == 0 || score > best_score){
                            best_score = score;
                        }
                    }
                }
                float weight = val == 1 ? 0.9f : 0.1f;
                avg_score += best_score * weight;
            }
        }
    }
    avg_score /= cnt;
    cache[k][b] = avg_score;
    return avg_score;
}

std::tuple<Board, int, int, int> infer_test(std::mt19937 & rng, int depth){
    Board b = 0;
    int total_score = 0;
    int total_steps = 0;
    while (true){
        b = fill_rand(b, rng);
        auto nexts = next_moves(b);
        float best_score = 0;
        Board best_b = 0;
        int best_move_score = 0;
        std::vector<std::unordered_map<Board, float>> cache(depth);
        for (int i = 0; i < 4; i ++){
            if (nexts[i].first != 0){
                float score = eval_kply(nexts[i].first, depth - 1, cache.data());// + nexts[i].second;
                if (best_b == 0 || score > best_score){
                    best_score = score;
                    best_b = nexts[i].first;
                    best_move_score = nexts[i].second;
                }
            }
        }

        total_score += best_move_score;
        total_steps += 1;
        if (best_b == 0){
            break;
        }
        b = best_b;
        int max_tile = 0;
        for (int i = 0; i < 16; i ++){
            int v = (b >> (i * 4)) & 0xF;
            if (v > max_tile){
                max_tile = v;
            }
        }
        if (max_tile >= 11){
            break;
        }
    }
    int max_tile = 0;
    for (int i = 0; i < 16; i ++){
        int v = (b >> (i * 4)) & 0xF;
        if (v > max_tile){
            max_tile = v;
        }
    }
    return {b, max_tile, total_score, total_steps};
}

std::array<int, 16> hist;
long long global_total_steps = 0;
volatile int global_step_count = 0;
std::mutex stats_mutex;
int total_eval_episodes = 1000000;
int search_depth = 3;
int report_every = 100;
unsigned int base_seed = 0;
bool seed_provided = false;

void thread_work(int thread_id){
    unsigned int seed = seed_provided ? base_seed + thread_id * 100 : thread_id * 100;
    std::mt19937 rng(seed);
    while (true){
        int step;
        if (true){
            std::lock_guard<std::mutex> lock(stats_mutex);
            step = global_step_count;
            if (step >= total_eval_episodes){
                break;
            }
            global_step_count ++;
        }

        auto [b, max_tile, total_score, total_steps] = infer_test(rng, search_depth);
        if (true){
            std::lock_guard<std::mutex> lock(stats_mutex);
            hist[max_tile] += 1;
            global_total_steps += total_steps;
            int total = 0;
            for (int i = 0; i < 16; i ++){
                total += hist[i];
            }
            if (report_every > 0 && total % report_every == 0){
                for (int j = 0; j < 16; j ++){
                    printf("%5d: %.3f\n", 1 << j, hist[j] * 100.0 / (total));
                }
                printf("total = %d\n", total);
            }
        }
    }
}

struct EvalConfig {
    std::string weights_path;
    int games = 1000000;
    int depth = 3;
    unsigned int seed = 0;
    bool has_seed = false;
    int report_every = 100;
};

void print_usage(const char * argv0){
    printf("Usage: %s weights_file\n", argv0);
    printf("       %s --weights PATH [options]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --help              Show this help message\n");
    printf("  --weights PATH      Raw float32 weights file\n");
    printf("  --games N           Number of games to evaluate (default: 1000000)\n");
    printf("  --depth N           Search depth, preserving old default depth=3\n");
    printf("  --seed INT          Base RNG seed; thread seeds are seed + thread_id * 100\n");
    printf("  --report-every N    Progress report interval, 0 disables progress (default: 100)\n");
}

bool parse_int(const char * text, int * out){
    char * end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L){
        return false;
    }
    *out = (int)value;
    return true;
}

bool parse_uint(const char * text, unsigned int * out){
    char * end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > 4294967295UL){
        return false;
    }
    *out = (unsigned int)value;
    return true;
}

bool parse_args(int argc, char ** argv, EvalConfig * config){
    for (int i = 1; i < argc; i ++){
        const char * arg = argv[i];
        if (std::strcmp(arg, "--help") == 0){
            print_usage(argv[0]);
            std::exit(0);
        }else if (std::strcmp(arg, "--weights") == 0 && i + 1 < argc){
            config->weights_path = argv[++i];
        }else if (std::strcmp(arg, "--games") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &config->games)) return false;
        }else if (std::strcmp(arg, "--depth") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &config->depth)) return false;
        }else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc){
            if (!parse_uint(argv[++i], &config->seed)) return false;
            config->has_seed = true;
        }else if (std::strcmp(arg, "--report-every") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &config->report_every)) return false;
        }else if (arg[0] != '-' && config->weights_path.empty()){
            config->weights_path = arg;
        }else{
            return false;
        }
    }
    return !config->weights_path.empty() && config->games >= 0 && config->depth > 0 && config->report_every >= 0;
}

int main(int argc, char ** argv) {
    EvalConfig config;
    if (!parse_args(argc, argv, &config)){
        fprintf(stderr, "invalid arguments\n");
        print_usage(argv[0]);
        return 1;
    }
    fill_move_lut();
    FILE * fin = fopen(config.weights_path.c_str(), "rb");
    if (!fin){
        fprintf(stderr, "failed to open weights file %s\n", config.weights_path.c_str());
        return 1;
    }
    size_t read = fread(network_weights, sizeof(network_weights), 1, fin);
    int extra = fgetc(fin);
    fclose(fin);
    if (read != 1 || extra != EOF){
        fprintf(stderr, "weights file must be exactly %zu bytes: %s\n", sizeof(network_weights), config.weights_path.c_str());
        return 1;
    }

    total_eval_episodes = config.games;
    search_depth = config.depth;
    report_every = config.report_every;
    base_seed = config.seed;
    seed_provided = config.has_seed;

    auto start = std::chrono::steady_clock::now();

    const int nthreads = 16;
    std::thread threads[nthreads];
    for (int i = 0; i < nthreads; i ++){
        threads[i] = std::thread(thread_work, i);
    }
    for (int i = 0; i < nthreads; i ++){
        threads[i].join();
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    for (int i = 0; i < 16; i ++){
        printf("%5d: %5d\n", 1 << i, hist[i]);
    }
    int wins = 0;
    int games = 0;
    for (int i = 0; i < 16; i ++){
        games += hist[i];
        if (i >= 11) wins += hist[i];
    }
    int failures = games - wins;
    double win_rate = games ? (double)wins / games : 0.0;
    double games_per_sec = elapsed > 0 ? games / elapsed : 0.0;
    double avg_steps = games ? (double)global_total_steps / games : 0.0;
    printf("games: %d\n", games);
    printf("wins: %d\n", wins);
    printf("failures: %d\n", failures);
    printf("win_rate: %.9f\n", win_rate);
    printf("elapsed_seconds: %.6f\n", elapsed);
    printf("games_per_sec: %.6f\n", games_per_sec);
    printf("avg_steps: %.6f\n", avg_steps);
    printf("hist_json: {");
    for (int i = 0; i < 16; i ++){
        printf("%s\"%d\":%d", i ? "," : "", 1 << i, hist[i]);
    }
    printf("}\n");
    return 0;
}

/*
    1:     0
    2:     0
    4:     0
    8:     0
   16:     0
   32:     0
   64:     0
  128:     0
  256:     3
  512:     0
 1024:     7
 2048: 99990
 4096:     0
 8192:     0
16384:     0
32768:     0
*/
