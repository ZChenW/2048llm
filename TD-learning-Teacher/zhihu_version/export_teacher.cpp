#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

typedef unsigned long long Board;

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

Board fill_rand(Board b, std::mt19937 &rng){
    int s = 0;
    for (int i = 0; i < 16; i ++){
        if (((b >> (i * 4)) & 0xF) == 0) s += 1;
    }
    if (s == 0) return b;
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
                        float score = eval_kply(nexts[j].first, k - 1, cache);
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
    if (cnt > 0) avg_score /= cnt;
    cache[k][b] = avg_score;
    return avg_score;
}

const char * ACTION_NAMES[4] = {"left", "right", "up", "down"};
const int JSON_ACTION_ORDER[4] = {2, 3, 0, 1};

struct Config {
    std::string weights_path;
    std::string out_path;
    int samples = 1000;
    int depth = 1;
    unsigned int seed = 1;
    int max_games = 10000;
    int min_max_tile = 0;
    double hard_state_ratio = 0.30;
    int report_every = 1000;
};

struct ActionScore {
    int action = 0;
    std::string name;
    Board afterstate = 0;
    int reward = 0;
    double score = 0;
};

struct Sample {
    Board board = 0;
    std::vector<ActionScore> actions;
    int max_tile = 0;
    int empty_cells = 0;
    int legal_move_count = 0;
    double top1_score = 0;
    double top2_score = 0;
    double score_margin = 0;
    bool hard = false;
};

struct Summary {
    long long count = 0;
    std::map<int, long long> max_tile_hist;
    std::map<int, long long> empty_cells_hist;
    std::map<int, long long> legal_move_count_hist;
    std::map<std::string, long long> action_dist;
    std::map<std::string, long long> score_margin_dist;
    long long hard_count = 0;
};

void usage(const char * argv0){
    std::cout
        << "Usage: " << argv0 << " --weights PATH --samples N --depth N --out PATH [options]\n\n"
        << "Options:\n"
        << "  --help                    Show this help message\n"
        << "  --weights PATH            Raw float32 weights file\n"
        << "  --samples N               Number of JSONL samples to export\n"
        << "  --depth N                 Teacher search depth, supported values 1 or 2\n"
        << "  --seed INT                RNG seed\n"
        << "  --out PATH                Output JSONL path\n"
        << "  --max-games N             Maximum self-play games to scan\n"
        << "  --min-max-tile N          Minimum max tile value for sampled boards\n"
        << "  --hard-state-ratio FLOAT  Target minimum fraction of hard/ambiguous states\n"
        << "  --report-every N          Progress report interval, 0 disables progress\n";
}

bool parse_int(const char * text, int * out){
    char * end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) return false;
    *out = (int)value;
    return true;
}

bool parse_uint(const char * text, unsigned int * out){
    char * end = nullptr;
    unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > 4294967295UL) return false;
    *out = (unsigned int)value;
    return true;
}

bool parse_double(const char * text, double * out){
    char * end = nullptr;
    double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) return false;
    *out = value;
    return true;
}

bool parse_args(int argc, char ** argv, Config * cfg){
    for (int i = 1; i < argc; i ++){
        const char * arg = argv[i];
        if (std::strcmp(arg, "--help") == 0){
            usage(argv[0]);
            std::exit(0);
        }else if (std::strcmp(arg, "--weights") == 0 && i + 1 < argc){
            cfg->weights_path = argv[++i];
        }else if (std::strcmp(arg, "--samples") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &cfg->samples)) return false;
        }else if (std::strcmp(arg, "--depth") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &cfg->depth)) return false;
        }else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc){
            if (!parse_uint(argv[++i], &cfg->seed)) return false;
        }else if (std::strcmp(arg, "--out") == 0 && i + 1 < argc){
            cfg->out_path = argv[++i];
        }else if (std::strcmp(arg, "--max-games") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &cfg->max_games)) return false;
        }else if (std::strcmp(arg, "--min-max-tile") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &cfg->min_max_tile)) return false;
        }else if (std::strcmp(arg, "--hard-state-ratio") == 0 && i + 1 < argc){
            if (!parse_double(argv[++i], &cfg->hard_state_ratio)) return false;
        }else if (std::strcmp(arg, "--report-every") == 0 && i + 1 < argc){
            if (!parse_int(argv[++i], &cfg->report_every)) return false;
        }else{
            return false;
        }
    }
    return !cfg->weights_path.empty()
        && !cfg->out_path.empty()
        && cfg->samples > 0
        && (cfg->depth == 1 || cfg->depth == 2)
        && cfg->max_games > 0
        && cfg->hard_state_ratio >= 0.0
        && cfg->hard_state_ratio <= 1.0;
}

int tile_exp_at(Board b, int pos){
    return (int)((b >> (pos * 4)) & 0xF);
}

int tile_value_from_exp(int exp){
    return exp == 0 ? 0 : (1 << exp);
}

int max_tile_value(Board b){
    int max_exp = 0;
    for (int i = 0; i < 16; i ++){
        max_exp = std::max(max_exp, tile_exp_at(b, i));
    }
    return tile_value_from_exp(max_exp);
}

int empty_count(Board b){
    int cnt = 0;
    for (int i = 0; i < 16; i ++){
        if (tile_exp_at(b, i) == 0) cnt += 1;
    }
    return cnt;
}

std::vector<ActionScore> compute_action_scores(Board b, int depth){
    std::vector<ActionScore> result;
    auto nexts = next_moves(b);
    std::vector<std::unordered_map<Board, float>> cache(depth);
    for (int i = 0; i < 4; i ++){
        if (nexts[i].first == 0) continue;
        double score = eval_kply(nexts[i].first, depth - 1, cache.data());
        if (depth == 1){
            score += nexts[i].second;
        }
        result.push_back({i, ACTION_NAMES[i], nexts[i].first, nexts[i].second, score});
    }
    std::sort(result.begin(), result.end(), [](const ActionScore & a, const ActionScore & b){
        if (a.score != b.score) return a.score > b.score;
        return a.action < b.action;
    });
    return result;
}

Sample make_sample(Board b, int depth){
    Sample sample;
    sample.board = b;
    sample.actions = compute_action_scores(b, depth);
    sample.max_tile = max_tile_value(b);
    sample.empty_cells = empty_count(b);
    sample.legal_move_count = (int)sample.actions.size();
    if (!sample.actions.empty()){
        sample.top1_score = sample.actions[0].score;
        sample.top2_score = sample.actions.size() >= 2 ? sample.actions[1].score : sample.actions[0].score;
        sample.score_margin = sample.top1_score - sample.top2_score;
    }
    double ambiguity_threshold = depth == 1 ? 32.0 : 0.0025;
    sample.hard = sample.empty_cells <= 4 || sample.legal_move_count <= 2 || sample.score_margin <= ambiguity_threshold;
    return sample;
}

bool should_keep_sample(const Sample & sample, const Config & cfg, int kept, int hard_kept, std::mt19937 & rng){
    if (sample.legal_move_count == 0) return false;
    if (sample.max_tile < cfg.min_max_tile) return false;
    int target_hard = (int)std::ceil(cfg.samples * cfg.hard_state_ratio);
    int remaining = cfg.samples - kept;
    int hard_needed = std::max(0, target_hard - hard_kept);
    if (sample.hard) return true;
    if (remaining <= hard_needed) return false;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return hard_kept >= target_hard || dist(rng) < 0.20;
}

std::string json_escape(const std::string & s){
    std::ostringstream out;
    for (char c : s){
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

void write_board_json(std::ostream & out, Board b){
    out << "[";
    for (int r = 0; r < 4; r ++){
        if (r) out << ",";
        out << "[";
        for (int c = 0; c < 4; c ++){
            if (c) out << ",";
            int exp = tile_exp_at(b, r * 4 + c);
            out << tile_value_from_exp(exp);
        }
        out << "]";
    }
    out << "]";
}

void write_sample_json(std::ostream & out, const Sample & sample, int depth){
    out << std::setprecision(9);
    out << "{\"board\":";
    write_board_json(out, sample.board);
    out << ",\"valid_moves\":[";
    bool first_valid = true;
    for (int ordered_action : JSON_ACTION_ORDER){
        for (const auto & action : sample.actions){
            if (action.action != ordered_action) continue;
            if (!first_valid) out << ",";
            first_valid = false;
            out << "\"" << action.name << "\"";
        }
    }
    out << "],\"action_scores\":{";
    bool first_score = true;
    for (int ordered_action : JSON_ACTION_ORDER){
        for (const auto & action : sample.actions){
            if (action.action != ordered_action) continue;
            if (!first_score) out << ",";
            first_score = false;
            out << "\"" << action.name << "\":" << action.score;
        }
    }
    out << "},\"action_ranking\":[";
    for (size_t i = 0; i < sample.actions.size(); i ++){
        if (i) out << ",";
        out << "\"" << sample.actions[i].name << "\"";
    }
    out << "]";
    out << ",\"teacher_action\":\"" << (sample.actions.empty() ? "" : sample.actions[0].name) << "\"";
    out << ",\"max_tile\":" << sample.max_tile;
    out << ",\"empty_cells\":" << sample.empty_cells;
    out << ",\"legal_move_count\":" << sample.legal_move_count;
    out << ",\"top1_score\":" << sample.top1_score;
    out << ",\"top2_score\":" << sample.top2_score;
    out << ",\"score_margin\":" << sample.score_margin;
    out << ",\"search_depth\":" << depth;
    out << ",\"source\":\"best_highwin_td\"";
    out << "}\n";
}

std::string margin_bucket(double margin){
    if (margin < 1e-9) return "0";
    if (margin < 0.001) return "(0,0.001)";
    if (margin < 0.01) return "[0.001,0.01)";
    if (margin < 0.1) return "[0.01,0.1)";
    if (margin < 1.0) return "[0.1,1)";
    if (margin < 10.0) return "[1,10)";
    if (margin < 100.0) return "[10,100)";
    if (margin < 1000.0) return "[100,1000)";
    return ">=1000";
}

void update_summary(Summary * summary, const Sample & sample){
    summary->count += 1;
    summary->max_tile_hist[sample.max_tile] += 1;
    summary->empty_cells_hist[sample.empty_cells] += 1;
    summary->legal_move_count_hist[sample.legal_move_count] += 1;
    if (!sample.actions.empty()) summary->action_dist[sample.actions[0].name] += 1;
    summary->score_margin_dist[margin_bucket(sample.score_margin)] += 1;
    if (sample.hard) summary->hard_count += 1;
}

template <typename K>
void write_map_json(std::ostream & out, const std::map<K, long long> & m){
    out << "{";
    bool first = true;
    for (const auto & [key, value] : m){
        if (!first) out << ",";
        first = false;
        out << "\"" << key << "\":" << value;
    }
    out << "}";
}

void write_summary(const std::string & out_path, const Summary & summary, const Config & cfg, int games_seen, double elapsed){
    std::ofstream out(out_path + ".summary.json");
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"count\": " << summary.count << ",\n";
    out << "  \"games_seen\": " << games_seen << ",\n";
    out << "  \"search_depth\": " << cfg.depth << ",\n";
    out << "  \"seed\": " << cfg.seed << ",\n";
    out << "  \"weights\": \"" << json_escape(cfg.weights_path) << "\",\n";
    out << "  \"source\": \"best_highwin_td\",\n";
    out << "  \"hard_count\": " << summary.hard_count << ",\n";
    out << "  \"hard_ratio\": " << (summary.count ? (double)summary.hard_count / summary.count : 0.0) << ",\n";
    out << "  \"elapsed_seconds\": " << elapsed << ",\n";
    out << "  \"max_tile_histogram\": ";
    write_map_json(out, summary.max_tile_hist);
    out << ",\n  \"empty_cells_histogram\": ";
    write_map_json(out, summary.empty_cells_hist);
    out << ",\n  \"legal_move_count_histogram\": ";
    write_map_json(out, summary.legal_move_count_hist);
    out << ",\n  \"action_distribution\": ";
    write_map_json(out, summary.action_dist);
    out << ",\n  \"score_margin_distribution\": ";
    write_map_json(out, summary.score_margin_dist);
    out << "\n}\n";
}

bool load_weights(const std::string & path){
    FILE * fin = fopen(path.c_str(), "rb");
    if (!fin){
        fprintf(stderr, "failed to open weights file %s\n", path.c_str());
        return false;
    }
    size_t read = fread(network_weights, sizeof(network_weights), 1, fin);
    int extra = fgetc(fin);
    fclose(fin);
    if (read != 1 || extra != EOF){
        fprintf(stderr, "weights file must be exactly %zu bytes: %s\n", sizeof(network_weights), path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char ** argv){
    Config cfg;
    if (!parse_args(argc, argv, &cfg)){
        fprintf(stderr, "invalid arguments\n");
        usage(argv[0]);
        return 1;
    }
    fill_move_lut();
    if (!load_weights(cfg.weights_path)) return 1;

    std::ofstream out(cfg.out_path);
    if (!out){
        fprintf(stderr, "failed to open output %s\n", cfg.out_path.c_str());
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    std::mt19937 rng(cfg.seed);
    Summary summary;
    int games_seen = 0;
    int kept = 0;
    int hard_kept = 0;

    for (; games_seen < cfg.max_games && kept < cfg.samples; games_seen ++){
        Board b = 0;
        b = fill_rand(b, rng);
        b = fill_rand(b, rng);
        while (kept < cfg.samples){
            Sample sample = make_sample(b, cfg.depth);
            if (sample.legal_move_count == 0) break;
            if (should_keep_sample(sample, cfg, kept, hard_kept, rng)){
                write_sample_json(out, sample, cfg.depth);
                update_summary(&summary, sample);
                kept += 1;
                if (sample.hard) hard_kept += 1;
                if (cfg.report_every > 0 && kept % cfg.report_every == 0){
                    std::cerr << "samples=" << kept << " games=" << (games_seen + 1)
                              << " hard_ratio=" << (kept ? (double)hard_kept / kept : 0.0) << "\n";
                }
            }
            Board next_afterstate = sample.actions[0].afterstate;
            if (max_tile_value(next_afterstate) >= 8192) break;
            b = fill_rand(next_afterstate, rng);
        }
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    write_summary(cfg.out_path, summary, cfg, games_seen, elapsed);

    std::cerr << "exported_samples: " << kept << "\n";
    std::cerr << "games_seen: " << games_seen << "\n";
    std::cerr << "elapsed_seconds: " << elapsed << "\n";
    if (kept < cfg.samples){
        std::cerr << "warning: requested " << cfg.samples << " samples but exported " << kept
                  << " before max-games limit\n";
        return 2;
    }
    return 0;
}
