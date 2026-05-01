#include <cstdio>
#include <random>
#include <array>
#include <mutex>
#include <atomic>
#include <thread>
#include <map>
#include <unordered_map>
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
                        float score = eval_kply(nexts[j].first, k - 1, cache) + nexts[j].second;
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

std::tuple<Board, int, int, int> infer_test(std::mt19937 & rng){
    Board b = 0;
    int total_score = 0;
    int total_steps = 0;
    const int k_max = 3;
    while (true){
        b = fill_rand(b, rng);
        auto nexts = next_moves(b);
        float best_score = 0;
        Board best_b = 0;
        int best_move_score = 0;
        std::unordered_map<Board, float> cache[k_max];
        for (int i = 0; i < 4; i ++){
            if (nexts[i].first != 0){
                float score = eval_kply(nexts[i].first, k_max - 1, cache) + nexts[i].second;
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

int main(int argc, char ** argv) {
    if (argc < 2){
        printf("Usage: %s weights_file\n", argv[0]);
        return 1;
    }
    fill_move_lut();
    FILE * fin = fopen(argv[1], "rb");
    fread(network_weights, sizeof(network_weights), 1, fin);
    fclose(fin);
    std::array<int, 16> hist;
    std::mt19937 rng(100);
    double avg_score = 0;
    double avg_steps = 0;
    const int nrun = 1000;
    for (int i = 0; i < nrun; i ++){
        auto [b, max_tile, total_score, total_steps] = infer_test(rng);
        hist[max_tile] += 1;
        avg_score += total_score;
        avg_steps += total_steps;
        printf("i = %d score %.3f steps %.3f    %c", i, avg_score / (i + 1), avg_steps / (i + 1), '\r');
        fflush(stdout);
        if (i >= nrun - 5){
            printf("i %d max tile: %d score: %d steps: %d      \n", i, 1 << max_tile, total_score, total_steps);
            display_board(b);
        }
        if ((i + 1) % 100 == 0){
            for (int j = 0; j < 16; j ++){
                printf("%5d: %.2f\n", 1 << j, hist[j] * 100.0 / (i + 1));
            }
            printf("avg score: %.2f avg steps: %.2f\n", avg_score / (1 + i), avg_steps / (1 + i));
        }
    }
    printf("\n");
    for (int i = 0; i < 16; i ++){
        printf("%5d: %5d\n", 1 << i, hist[i]);
    }
    printf("avg score: %.2f avg steps: %.2f\n", avg_score / nrun, avg_steps / nrun);
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
  256:     0
  512:     1
 1024:     0
 2048:     0
 4096:     7
 8192:    63
16384:   571
32768:   358
avg score: 393447.10 avg steps: 14354.36
*/
