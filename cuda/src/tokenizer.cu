#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// .tok file format:
//   [4 bytes]  n  (uint32, number of tokens)
//   [n*4 bytes] token_ids (uint32 each)
// ============================================================================

static bool read_tok_file(const char* path,
                          std::vector<int>& out_ids) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[tokenizer] Cannot open '%s'\n", path);
        return false;
    }
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n == 0 || n > 65536) {
        fprintf(stderr, "[tokenizer] Bad token count in '%s'\n", path);
        fclose(f); return false;
    }
    out_ids.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = 0;
        if (fread(&id, 4, 1, f) != 1) {
            fprintf(stderr, "[tokenizer] Truncated token file '%s'\n", path);
            fclose(f); return false;
        }
        out_ids[i] = (int)id;
    }
    fclose(f);
    return true;
}

static bool write_tok_file(const char* path,
                           const int* ids, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[tokenizer] Cannot write '%s'\n", path);
        return false;
    }
    uint32_t un = (uint32_t)n;
    fwrite(&un, 4, 1, f);
    for (int i = 0; i < n; i++) {
        uint32_t uid = (uint32_t)ids[i];
        fwrite(&uid, 4, 1, f);
    }
    fclose(f);
    return true;
}

// ============================================================================
// Tokenizer member functions
// ============================================================================

bool Tokenizer::load(const char* path) {
    return read_tok_file(path, ids);
}

bool Tokenizer::save(const char* path, const int* tok_ids, int n) const {
    return write_tok_file(path, tok_ids, n);
}

// ============================================================================
// C helpers for use from main.cu without STL
// ============================================================================

int* tok_load_alloc(const char* path, int* out_n) {
    std::vector<int> v;
    if (!read_tok_file(path, v)) return nullptr;
    int n = (int)v.size();
    int* buf = (int*)malloc((size_t)n * sizeof(int));
    if (!buf) return nullptr;
    memcpy(buf, v.data(), (size_t)n * sizeof(int));
    *out_n = n;
    return buf;
}

bool tok_save(const char* path, const int* ids, int n) {
    return write_tok_file(path, ids, n);
}
