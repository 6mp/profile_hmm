# Profile HMM Viterbi Decoder (C++)

A high-performance C++ implementation of the Viterbi algorithm for decoding biological sequences using a profile Hidden Markov Model (HMM). This project emphasizes correctness, efficiency, and clean memory layout through tight structuring and modern C++ features.

---

## Key Features

- **One-pass Viterbi decoder** using log-space dynamic programming  
- **Efficient DP matrix layout** using a 1D vector with contiguous match/insert/delete state grouping per cell  
- **Pointer-based backtracking** to reconstruct the optimal path with minimal overhead  
- **Robust FASTA and HMMER3 parsing**, with error checking and minimal dependencies  
- **Meson build system** for fast and portable configuration

---

## Build Instructions

This project uses Meson for build configuration:

```bash
meson setup build
meson compile -C build
```

You’ll get a binary named `viterbi_decoder` in `build/`.

---

## Usage

```bash
./build/viterbi_decoder model.hmm queries.fasta
```

Optionally, a third argument can be provided to redirect output to a file:

```bash
./build/viterbi_decoder model.hmm queries.fasta output.txt
```

---

## Input Formats

- **Model:** Plaintext profile HMM in HMMER3 format (emissions, transitions, and alphabet are parsed)
- **Queries:** FASTA format, one or more sequences

---

## Output Format

Each line corresponds to a query:

```
<sequence_id> <log_likelihood_score> <aligned_sequence>
```

Alignment characters:
- `-`: deletion (D state)
- uppercase: match (M state)
- lowercase: insertion (I state)

---

## Notable Implementation Details

### DP Matrix Structure

- The DP matrix is a 1D vector of `DPCell` structs arranged in groups of 3 (M, I, D) per (j, i) coordinate.
- Layout:  
  ```
  [(M₀₀, I₀₀, D₀₀), (M₀₁, I₀₁, D₀₁), ..., (M₁₀, I₁₀, D₁₀), ...]
  ```
- Efficient indexing:  
  ```cpp
  index = ((j * num_columns) + i) * 3 + state_index;
  ```

### `DPCell` Backtracking

Each `DPCell` holds:
- The score of the best path to that cell
- A `prev` pointer to the previous cell in the path
- The current state and aligned character (for backtrace reconstruction)

This allows pointer-based backtracking without additional path tracking structures.

### Probabilities in Log-Space

All transition and emission probabilities are stored as negative log values (converted during parsing). This avoids underflow and simplifies Viterbi scoring.

### State Transition Lookup

The key `"MI"`, `"II"`, `"DM"`, etc. is dynamically generated using:

```cpp
std::string getTransToKey(State to) const
```

This drives all transition lookups from state `s → t`.

### DefaultDouble Wrapper

Custom wrapper around `double`:
- Automatically initializes to `-inf`
- Enables safe map access without pre-checking for key existence

```cpp
std::unordered_map<char, DefaultDouble> m_eI[];
```

### FASTA Parsing

- Deduplicates sequence IDs
- Enforces non-empty sequences
- Supports arbitrary ordering

---

## Extensibility

This implementation was written to be cleanly extensible. Potential additions:

- Parallel Viterbi decoding for batch FASTA files
- Support for HMM emitters beyond nucleotides (e.g., protein models)
- Beam search for approximate decoding
- JSON output format

---

## License

MIT License. See `LICENSE` file.
