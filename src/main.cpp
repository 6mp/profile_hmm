#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <fstream>
#include <string>
#include <ranges>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <sstream>

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

struct BetterDouble {
    BetterDouble(double value)
        : value(value) {};
    BetterDouble()
        : value(NEG_INF) {};
    operator double() const { return value; }
    operator double&() { return value; }
    /*    auto operator=(double value) -> BetterDouble& {
            this->value = value;
            return *this;
        }
        auto operator=(BetterDouble value) -> BetterDouble& {
            this->value = value.value;
            return *this;
        }
            auto operator+=(double value) -> BetterDouble& {
                    this->value += value;
                    return *this;
            }
            auto operator+=(BetterDouble value) -> BetterDouble& {
                    this->value += value.value;
                    return *this;
            }
            auto operator+(double value) -> BetterDouble { return BetterDouble(this->value + value); }

            auto operator+(BetterDouble value) -> BetterDouble { return BetterDouble(this->value + value.value); }

            auto operator-(double value) -> BetterDouble { return BetterDouble(this->value - value); }

            auto operator-(BetterDouble value) -> BetterDouble { return BetterDouble(this->value - value.value); }*/
    double value;
};

enum class State : std::size_t { M = 0, I = 1, D = 2 };

constexpr auto state_to_symbol(State s) -> char { return (s == State::M) ? 'M' : (s == State::I) ? 'I' : 'D'; };

// put the larger things first to minimize padding
struct DPCell {
    double score = NEG_INF;
    const DPCell* prev = nullptr;    // pointer to previous DPCell for backtracking
    State state = State::M;
    char alignedChar = '\0';    // char for building alignment string

    [[nodiscard]] auto getTransToKey(State to) const -> std::string {
        return {state_to_symbol(state), state_to_symbol(to)};
    }
};

// memory layout for cache, reason why is to compute the
// next state its easy to look back since its all contiguous
// [m_0, i_0, d_0, m_1, i_1, d_1, m_2, i_2, d_2, ...]

// if it wasnt 1d it would look like this
// [[m_0, i_0, d_0], [m_1, i_1, d_1], [m_2, i_2, d_2], ...]
class DPMatrix {
public:
    // Storing it all in a 1d array
    DPMatrix(const std::size_t rows, const std::size_t cols)
        : m_rows(rows)
        , m_cols(cols)
        , data(rows * cols * 3) {}

    auto operator()(const std::size_t j, const std::size_t i, const State s) -> DPCell& { return data[index(j, i, s)]; }
    const DPCell& operator()(const std::size_t j, const std::size_t i, const State s) const {
        return data[index(j, i, s)];
    }

private:
    [[nodiscard]] std::size_t index(std::size_t j, std::size_t i, State s) const {
        // Each cell (j, i) holds three DPCell objects in order: M, I, D.
        // since the state is an enum with number assigned 0, 1, 2 it can be used to access each cell
        return ((j * m_cols) + i) * 3 + static_cast<std::size_t>(s);
    }
    std::size_t m_rows;    // dont need to acc store this
    std::size_t m_cols;
    std::vector<DPCell> data;
};

// might use this later
class Transition {
public:
    State from;
    State to;
    double score;
};

//
// The HMM class stores the model parameters and implements the Viterbi algorithm.
// The DP table is allocated as one contiguous array via DPMatrix.
// Note: The emission and transition probabilities here are stored in
// vectors of unordered_maps. In a real system they would be read from a model file.
//
class HMM {
public:
    explicit HMM(const std::string& path) {
        //        this->m_eM.emplace_back(std::pair<char, double>{'k', .1});
        load(path);
    }

    std::size_t m_nStates{};           // number of match states (K)
    std::vector<char> m_alphabet{};    // e.g. {"A", "C", "G", "T"}
    // Emission probabilities for insertions; size: m_nStates+1.
    std::vector<std::unordered_map<char, double>> m_eI{};
    // Emission probabilities for match states; size: m_nStates+1 (index 0 may be unused)
    std::vector<std::unordered_map<char, double>> m_eM{};
    // Transition probabilities; one unordered_map per row (0..m_nStates).
    // Keys are two-letter strings (e.g. "MI", "II", "DD", etc.).
    std::vector<std::unordered_map<std::string, BetterDouble>> m_t{};

    void load(const std::string& path) {
        std::ifstream fin(path);
        if (!fin) {
            throw std::runtime_error("Unable to open file: " + path);
        }

        // equal to this: stream = line.strip().split()
        auto split = [](const std::string& s) {
            std::vector<std::string> tokens;
            std::istringstream iss(s);
            std::copy(
                std::istream_iterator<std::string>(iss),
                std::istream_iterator<std::string>(),
                std::back_inserter(tokens));
            return tokens;
        };

        std::vector<std::string> trans_order;
        std::string line;
        while (std::getline(fin, line)) {
            const auto tokens = split(line);

            // read len
            if (tokens[0] == "LENG") {
                m_nStates = std::stoul(tokens[1]);
                m_eI.resize(m_nStates + 1);
                m_eM.resize(m_nStates + 1);
                m_t.resize(m_nStates + 1);
            }

            if (tokens[0] == "HMM") {
                // read alphabest stream[1:]
                m_alphabet = std::vector(tokens.begin() + 1, tokens.end()) |
                             std::views::transform([](const auto& s) { return s[0]; }) |
                             std::ranges::to<std::vector<char>>();

                // read transition order
                std::getline(fin, line);
                const auto split_trans_order = split(line);

                trans_order = split_trans_order | std::views::transform([](const auto& x) {
                                  return std::string{
                                      static_cast<char>(std::toupper(x[0])), static_cast<char>(std::toupper(x[3]))};
                              }) |
                              std::ranges::to<std::vector<std::string>>();

                // read the next line, if it is the COMPO line then ignore it and read one more line
                std::getline(fin, line);
                const auto split_compo = split(line);
                if (split_compo[0] == "COMPO") {
                    std::getline(fin, line);
                }

                // # now the stream should be at the I0 state; read the emission of the I0 state
                for (const auto [x, y] : std::views::zip(m_alphabet, split(line))) {
                    m_eI[0][x] = y == "*" ? NEG_INF : -std::stod(y);
                }

                // now the stream should be at the B state; read in the transition probability
                std::getline(fin, line);
                for (const auto& [x, y] : std::views::zip(trans_order, split(line))) {
                    m_t[0][x] = y == "*" ? NEG_INF : -std::stod(y);
                }

                break;
            }
        }

        for (std::size_t idx = 1; idx < m_nStates + 1; ++idx) {
            // read each set of 3 lines at a time
            std::getline(fin, line);
            const auto split_em = split(line);
            if (std::stod(split_em[0]) != idx) {
                throw std::runtime_error("Expected state index " + std::to_string(idx) + " but got " + split_em[0]);
            }

            for (const auto& [x, y] : std::views::zip(m_alphabet, split_em | std::views::drop(1))) {
                m_eM[idx][x] = y == "*" ? NEG_INF : -std::stod(y);
            }

            std::getline(fin, line);
            for (const auto& [x, y] : std::views::zip(m_alphabet, split(line))) {
                m_eI[idx][x] = y == "*" ? NEG_INF : -std::stod(y);
            }

            std::getline(fin, line);

            for (const auto& [x, y] : std::views::zip(trans_order, split(line))) {
                m_t[idx][x] = y == "*" ? NEG_INF : -std::stod(y);
            }
        }
    }

    // Given a previous cell (j, i), find the best transition to state 'to'.
    // Returns the best DPCell for state 'to'.
    // still have to add the emission cost and aligned char bc those are specific to the state
    [[nodiscard]] auto getBestTrans(const DPMatrix& dp, std::size_t j, std::size_t i, State to, char alignedChar)
        -> DPCell {
        constexpr std::array<State, 3> states{State::M, State::I, State::D};

        DPCell bestCell;
        bestCell.state = to;
        bestCell.alignedChar = alignedChar;

        for (const State s : states) {
            const auto& prevCell = dp(j, i, s);
            const auto transKey = prevCell.getTransToKey(to);

            double candidate = prevCell.score + m_t[j][transKey];
            if (candidate > bestCell.score) {
                bestCell.score = candidate;
                bestCell.prev = &prevCell;
            }
        }

        return bestCell;
    }

    // Viterbi returns a pair: best log-score and an alignment string.
    // In the alignment string, uppercase letters are matches, lowercase letters are insertions, and '-' is a
    // deletion.
    [[nodiscard]] auto viterbi(std::string_view query) -> std::pair<double, std::string> {
        const std::size_t L = query.size();
        const std::size_t K = m_nStates;

        // Create a DPMatrix with (K+1) rows and (L+1) columns.
        DPMatrix dp(K + 1, L + 1);

        // Base case: start at (j=0, i=0) in the Match state with score 0.
        dp(0, 0, State::M).score = 0.0;
        dp(0, 0, State::M).prev = nullptr;
        dp(0, 0, State::M).state = State::M;

        // Fill in the DP table.
        for (std::size_t j = 0; j <= K; ++j) {    // possible kernel in here to do like 3 at a time
            for (std::size_t i = 0; i <= L; ++i) {

                // --- Deletion (D) ---
                // Deletion: move from (j-1, i) to (j, i) without consuming a query char.
                if (j > 0) {
                    std::size_t prev_j = j - 1;
                    dp(j, i, State::D) = getBestTrans(dp, prev_j, i, State::D, '-');
                    // no emission cost to add in
                }

                // --- Insertion (I) ---
                // Insertion: move from (j, i-1) to (j, i) while consuming one query character.
                if (i > 0 && j <= K) {

                    std::size_t prev_i = i - 1;
                    auto thing = getBestTrans(dp, j, prev_i, State::I, std::tolower(query[prev_i]));

                    // Add emission cost for insertion from m_eI[j]
                    const auto letter = query[prev_i];

                    // will throw if letter doesnt exist, shouldnt happen
                    thing.score += m_eI[j].find(letter)->second;

                    dp(j, i, State::I) = thing;
                }

                // --- Match (M) ---
                // Match: move from (j-1, i-1) to (j, i) while consuming one query character.
                if (j > 0 && i > 0) {
                    std::size_t prev_j = j - 1;
                    std::size_t prev_i = i - 1;

                    // idk if i actually have to do toupper here cause query should be in all caps
                    auto thing = getBestTrans(dp, prev_j, prev_i, State::M, std::toupper(query[prev_i]));

                    // Add emission cost for match from m_eM[j]
                    const auto letter = query[prev_i];

                    // will throw if letter doesnt exist
                    thing.score += m_eM[j].find(letter)->second;

                    dp(j, i, State::M) = thing;
                }
            }
        }

        // --- Final transition ---

        // just null termin for aligned char it doesnt matter this is gonna get skipped in backtracking
        auto last = getBestTrans(dp, K, L, State::M, '\0');
        if (last.score == NEG_INF || last.prev == nullptr)
            return {NEG_INF, "$"};

        // just following pointer chain backwards
        std::string alignment;
        alignment.resize(L);        // alignment gonna be the same size as the query so pre alloc
        std::size_t pos = L - 1;    // inserting from end of string to front so no reverse needed
        for (const DPCell* cell = last.prev; cell->prev != nullptr; cell = cell->prev) {
            alignment[pos--] = cell->alignedChar;
        }
        return {last.score, alignment};
    }
};

int main() {

    HMM hmm1("../examples/test3/model.hmm");

    // Example query sequence.
    std::string query = "ACGTACG";
    auto [score, alignment] = hmm1.viterbi(query);
    std::cout << "Score: " << score << '\n';
    std::cout << "Alignment: " << alignment << '\n';
}