#pragma once
#include <cstdint>
#include <vector>

// ============================================================================
// Tokenizer
//
// The C++/CUDA runtime uses a minimal binary-file-based tokenizer:
//   - Encoding (text → token IDs) is handled by tools/tokenize.py using the
//     original HuggingFace tokenizer, which writes a .tok binary file.
//   - Decoding (token IDs → text) is similarly handled by tools/tokenize.py.
//
// Prompt binary format (.tok file):
//   [n_tokens : uint32]
//   [token_ids: n_tokens * uint32]
//
// This host-side struct just wraps the two I/O operations.
// ============================================================================

struct Tokenizer {
    // Load token IDs from a .tok binary file produced by tools/tokenize.py.
    // Returns true on success.
    bool load(const char* path);

    // Save token IDs to a .tok binary file (for later decoding by Python).
    bool save(const char* path, const int* ids, int n) const;

    // The loaded token sequence
    std::vector<int> ids;
};

// ============================================================================
// Convenience helpers
// ============================================================================

// Read token IDs from a .tok file into a plain C array.
// *out_n receives the count.  Caller must free() the returned buffer.
// Returns nullptr on failure.
int* tok_load_alloc(const char* path, int* out_n);

// Write token IDs to a .tok file.
bool tok_save(const char* path, const int* ids, int n);
