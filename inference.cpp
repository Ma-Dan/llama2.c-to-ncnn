#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "net.h"

const int vocab_size = 32000;

const float temp = 1, topp = 0.9;
const int topk = 300;

struct bpe {
    int max_token_length;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> lookup;
    std::vector<float> scores;

    void load(std::string path);
    std::vector<int> encode(std::string s);
};

void bpe::load(std::string path) {
    vocab.resize(vocab_size);
    scores.resize(vocab_size);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) exit(1);
    fread(&max_token_length, sizeof(int), 1, f);
    std::vector<char> s(max_token_length + 1);
    for (int i = 0; i < vocab_size; i++) {
        fread(scores.data() + i, sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        fread(s.data(), sizeof(char) * len, 1, f);
        s[len] = 0;
        vocab[i] = s.data();
        lookup[vocab[i]] = i;
    }
    fclose(f);
}

typedef struct {
    const char *str;
    int id;
} TokenIndex;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

std::vector<int> bpe::encode(std::string s) {
    if (s.length() && s[0] != ' ') s = " " + s;

    // sort vocabulary
    TokenIndex *sorted_vocab = (TokenIndex *)malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = vocab[i].c_str();
        sorted_vocab[i].id = i;
    }
    qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);

    char* str_buffer = (char *)malloc((s.length()*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    std::vector<int> tokens;
    for (const char *c = s.c_str(); *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(id);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens.push_back((unsigned char)str_buffer[i] + 3);
            }
        }
        str_len = 0;
    }

    while (true) {
        float best_score = -1e10;
        int best_index = -1, best_token = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            auto merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
            if (lookup.count(merged) && scores[lookup[merged]] > best_score) {
                best_score = scores[lookup[merged]];
                best_index = i;
                best_token = lookup[merged];
            }
        }

        if (best_token == -1) break;

        tokens[best_index] = best_token;
        tokens.erase(tokens.begin() + best_index + 1);
    }

    free(str_buffer);
    free(sorted_vocab);

    return tokens;
}

struct tinyllama {
    ncnn::Net net;
    std::vector<ncnn::Mat> kcache, vcache;
    int ctx_length, pos, n_l, dim, n_heads;
    std::vector<float> freqs_cos, freqs_sin;
    tinyllama(std::string bin, std::string param, int n_layers, int ctx_len,
              int dim_, int nh);
    std::vector<float> forward(int token);
};

tinyllama::tinyllama(std::string bin, std::string param, int n_layers,
                     int ctx_len, int dim_, int nh) {
    if (net.load_param(param.c_str())) exit(1);
    if (net.load_model(bin.c_str())) exit(1);
    pos = 0;
    n_l = n_layers;
    ctx_length = ctx_len;
    dim = dim_;
    n_heads = nh;
    kcache.resize(n_l);
    vcache.resize(n_l);
    int head_dim = dim / n_heads;
    freqs_cos.resize(ctx_length * head_dim / 2);
    freqs_sin.resize(ctx_length * head_dim / 2);

    for (int i = 0; i < ctx_length; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            auto x = i / pow(10000.0, j * 2 / (double)head_dim);
            freqs_cos[i * head_dim / 2 + j] = cos(x);
            freqs_sin[i * head_dim / 2 + j] = sin(x);
        }
    }

    for (int i = 0; i < n_l; i++) {
        kcache[i].create(dim, 0);
        vcache[i].create(dim, 0);
    }
    net.opt.use_fp16_storage = false;
    // net.opt.num_threads = 8;
}

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

std::vector<float> tinyllama::forward(int token) {
    ncnn::Mat x(1), fc, fs;
    *((int*)x) = token;

    int head_dim = dim / n_heads;
    fc.create(head_dim / 2, pos + 1);
    fs.create(head_dim / 2, pos + 1);
    for (int i = 0; i < (pos + 1) * head_dim / 2; i++) {
        fc[i] = freqs_cos[i];
        fs[i] = freqs_sin[i];
    }

    auto ex = net.create_extractor();
    ex.input("in", x);
    ex.input("freqs_cos", fc);
    ex.input("freqs_sin", fs);
    for (int i = 0; i < n_l; i++) {
        auto layer_name = std::to_string(i);
        auto kc_name = "kcache." + layer_name;
        auto vc_name = "vcache." + layer_name;
        ex.input(kc_name.c_str(), kcache[i]);
        ex.input(vc_name.c_str(), vcache[i]);
    }

    /*ncnn::Mat test0_mat;
    ex.extract("59", test0_mat);
    pretty_print(test0_mat);

    ncnn::Mat test1_mat;
    ex.extract("60", test1_mat);
    pretty_print(test1_mat);

    ncnn::Mat test2_mat;
    ex.extract("61", test2_mat);
    pretty_print(test2_mat);*/

    ncnn::Mat logits_mat;
    for (int i = 0; i < n_l; i++) {
        auto layer_name = std::to_string(i);
        auto kc_name = "kcache_out." + layer_name;
        auto vc_name = "vcache_out." + layer_name;
        ex.extract(kc_name.c_str(), kcache[i]);
        ex.extract(vc_name.c_str(), vcache[i]);
    }
    ex.extract("out", logits_mat);
    std::vector<float> logits(logits_mat.total());

    for (size_t i = 0; i < logits_mat.total(); i++) logits[i] = logits_mat[i];

    pos++;
    if (pos == ctx_length) {
        pos--;
        auto shift_cache = [&](ncnn::Mat& x) -> void {
            ncnn::Mat y;
            y.create(dim, ctx_length);
            for (int i = 0; i < dim * ctx_length; i++) {
                y[i] = x[i + dim];
            }
            x = y;
        };
        auto shift = [&](std::vector<ncnn::Mat>& v) -> void {
            for (auto& x : v) {
                shift_cache(x);
            }
        };
        shift(kcache);
        shift(vcache);
    }

    return logits;
}

int sample(const std::vector<float>& logits, float temp, float topp, int topk) {
    // return std::max_element(logits.begin(), logits.end()) - logits.begin();

    assert(logits.size() == vocab_size);

    if (fabsf(temp) < 1e-8)
        return std::max_element(logits.begin(), logits.end()) - logits.begin();

    static std::mt19937_64 rng(3407);  // haha
    static std::uniform_real_distribution<float> dist(0, 1);

    std::vector<std::pair<float, int>> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) probs[i] = {logits[i] / temp, i};
    std::sort(probs.begin(), probs.end(),
              std::greater<std::pair<float, int>>());
    while (probs.size() > topk) probs.pop_back();

    // softmax
    auto maximum = probs[0].first;
    std::transform(probs.begin(), probs.end(), probs.begin(),
                   [maximum](auto x) {
                       return std::make_pair(expf(x.first - maximum), x.second);
                   });
    auto sum = std::accumulate(probs.begin(), probs.end(), 0.0f,
                               [](auto x, auto y) { return x + y.first; });
    std::transform(probs.begin(), probs.end(), probs.begin(), [sum](auto x) {
        return std::make_pair(x.first / sum, x.second);
    });

    sum = 0;
    int last = 0;
    for (int i = 0; i < (int)probs.size(); i++) {
        sum += probs[i].first;
        last = i;
        if (sum > topp) break;
    }

    float r = dist(rng) * sum;
    sum = 0;
    for (int i = 0; i <= last; i++) {
        sum += probs[i].first;
        if (sum > r) return probs[i].second;
    }
    return probs[last].second;
}

// ./inference MODEL PROMPT OUT-TOKEN-COUNT
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT OUT-TOKEN-COUNT"
                  << std::endl;
        return 1;
    }

    std::string model_bin = argv[1], model_param = argv[1],
                tokenizer_path = "tokenizer.bin", prompt = argv[2];
    int token_count = std::stoi(argv[3]);
    model_bin += ".bin";
    model_param += ".param";

    int ctx_len, n_layers, dim, n_heads;
    std::ifstream desc(std::string(argv[1]) + ".desc");
    desc >> ctx_len >> n_layers >> dim >> n_heads;
    desc.close();

    tinyllama model(model_bin, model_param, n_layers, ctx_len, dim, n_heads);

    // tokenize prompt
    bpe tokenizer;
    tokenizer.load(tokenizer_path);

    auto tokens = tokenizer.encode(prompt);
    tokens.insert(tokens.begin(), 1);  // bos
    int prompt_end = tokens.size();
    tokens.resize(token_count);

    // for (int i = 0; i < token_count; i++) std::cout << tokens[i] << " ";
    // std::cout << std::endl;

    std::chrono::steady_clock clk;
    auto t0 = clk.now();

    // feed forward
    for (int i = 0; i < token_count; i++) {
        std::cout << tokenizer.vocab[tokens[i]] << std::flush;
        auto logits = model.forward(tokens[i]);
        if (i < prompt_end - 1) continue;
        tokens[i + 1] = sample(logits, temp, topp, topk);
        if (i == 0) t0 = clk.now();
    }
    std::cout << std::endl;

    auto t1 = clk.now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cerr << elapsed.count() / (token_count - 1) << " ms / token"
              << std::endl;

    exit(0);
}
