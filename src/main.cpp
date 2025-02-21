#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cassert>

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

enum class State : std::size_t { M = 0, I = 1, D = 2 };

constexpr auto state_to_symbol(State s) -> char { return (s == State::M) ? 'M' : (s == State::I) ? 'I' : 'D'; };

// put the larger things first to minimize padding
struct DPCell {
    double score = NEG_INF;
    DPCell* prev = nullptr;    // pointer to previous DPCell for backtracking
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
    std::size_t m_rows; // dont need to acc store this
    std::size_t m_cols;
    std::vector<DPCell> data;
};

//
// The HMM class stores the model parameters and implements the Viterbi algorithm.
// The DP table is allocated as one contiguous array via DPMatrix.
// Note: The emission and transition probabilities here are stored in
// vectors of unordered_maps. In a real system they would be read from a model file.
//
class HMM {
public:
    std::size_t m_nStates{};                  // number of match states (K)
    std::vector<std::string> m_alphabet{};    // e.g. {"A", "C", "G", "T"}
    // Emission probabilities for insertions; size: m_nStates+1.
    std::vector<std::unordered_map<std::string, double>> m_eI{};
    // Emission probabilities for match states; size: m_nStates+1 (index 0 may be unused)
    std::vector<std::unordered_map<std::string, double>> m_eM{};
    // Transition probabilities; one unordered_map per row (0..m_nStates).
    // Keys are two-letter strings (e.g. "MI", "II", "DD", etc.).
    std::vector<std::unordered_map<std::string, double>> m_t{};

    auto load(const std::string& path) {}

    // Given a previous cell (j, i, s), find the best transition to state 'to'.
    // Returns the best DPCell for state 'to'.
    // still have to add the emission cost and aligned char bc those are specific to the state
    auto getBestTrans(DPMatrix& dp, std::size_t j, std::size_t i, State to) const -> DPCell {
        constexpr std::array<State, 3> states{State::M, State::I, State::D};

        DPCell bestCell;
        bestCell.state = to;

        for (const State s : states) {
            auto& prevCell = dp(j, i, s);
            auto transKey = prevCell.getTransToKey(to);

            // make sure transition exists, if not skip bc -inf + anything is irrelevant
            if (auto it = m_t[j].find(transKey); it != m_t[j].end()) {
                double candidate = prevCell.score + it->second;
                if (candidate > bestCell.score) {
                    //                    bestCell = prevCell;
                    bestCell.score = candidate;
                    bestCell.prev = (&prevCell);
                }
            }
        }

        return bestCell;
    }

    // Viterbi returns a pair: best log-score and an alignment string.
    // In the alignment string, uppercase letters are matches, lowercase letters are insertions, and '-' is a
    // deletion.
    [[nodiscard]] auto viterbi(std::string_view query) const -> std::pair<double, std::string> {
        const std::size_t L = query.size();
        const std::size_t K = m_nStates;

        // Create a DPMatrix with (K+1) rows and (L+1) columns.
        DPMatrix dp(K + 1, L + 1);

        // Base case: start at (j=0, i=0) in the Match state with score 0.
        dp(0, 0, State::M).score = 0.0;
        dp(0, 0, State::M).prev = nullptr;
        dp(0, 0, State::M).state = State::M;

        // Fill in the DP table.
        for (std::size_t j = 0; j <= K; ++j) {
            for (std::size_t i = 0; i <= L; ++i) {

                // --- Deletion (D) ---
                // Deletion: move from (j-1, i) to (j, i) without consuming a query char.
                if (j > 0) {
                    /*
                                        double bestScore = NEG_INF;
                                        DPCell* bestPrev = nullptr;
                                        std::size_t prev_j = j - 1;
                                        for (State s : {State::M, State::I, State::D}) {
                                            auto& prevCell = dp(prev_j, i, s);
                                            // Transition key depends on previous state: e.g., from M, key "MD"
                                            std::string key = (s == State::M) ? "MD" : (s == State::I) ? "ID" : "DD";
                                            double trans = NEG_INF;
                                            if (auto it = m_t[prev_j].find(key); it != m_t[prev_j].end())
                                                trans = it->second;
                                            double candidate = prevCell.score + trans;
                                            if (candidate > bestScore) {
                                                bestScore = candidate;
                                                bestPrev = &prevCell;
                                            }
                                        }

                    */

                    std::size_t prev_j = j - 1;
                    auto thing = getBestTrans(dp, prev_j, i, State::D);
                    //                    assert(thing.score == bestScore);
                    //                    assert(thing.prev == bestPrev);

                    DPCell& cell = dp(j, i, State::D);
                    cell = thing;
                    cell.alignedChar = '-';    // deletion: output gap
                                               //                    cell.score = thing.score;
                                               //                    cell.prev = thing.prev;
                                               //                    cell.state = State::D;
                                               //                    cell.alignedChar = '-';    // deletion: output gap
                }

                // --- Insertion (I) ---
                // Insertion: move from (j, i-1) to (j, i) while consuming one query character.
                if (i > 0 && j <= K) {
                    /*            double bestScore = NEG_INF;
                                DPCell* bestPrev = nullptr;
                                for (State s : {State::M, State::I, State::D}) {
                                    const DPCell& prevCell = dp(j, i - 1, s);
                                    std::string key = (s == State::M) ? "MI" : (s == State::I) ? "II" : "DI";
                                    double trans = NEG_INF;
                                    if (auto it = m_t[j].find(key); it != m_t[j].end())
                                        trans = it->second;
                                    double candidate = prevCell.score + trans;
                                    if (candidate > bestScore) {
                                        bestScore = candidate;
                                        bestPrev = const_cast<DPCell*>(&prevCell);
                                    }
                                }*/

                    auto thing = getBestTrans(dp, j, i - 1, State::I);
                    /*                    assert(thing.score == bestScore);
                                        assert(thing.prev == bestPrev);*/

                    // Add emission cost for insertion from m_eI[j]
                    std::string letter(1, query[i - 1]);
                    double emission = NEG_INF;
                    if (auto it = m_eI[j].find(letter); it != m_eI[j].end())
                        emission = it->second;
                    //                    bestScore += emission;
                    thing.score += emission;

                    DPCell& cell = dp(j, i, State::I);

                    cell = thing;
                    cell.alignedChar = std::tolower(query[i - 1]);
                    //                    cell.score = thing.score;
                    //                    cell.prev = thing.prev;
                    //                    cell.state = State::I;
                    /*                    cell.score = bestScore;
                                        cell.prev = bestPrev;
                                        cell.state = State::I;
                                        // Insertion: output the query letter in lowercase.
                                        cell.alignedChar = std::tolower(query[i - 1]);*/
                }

                // --- Match (M) ---
                // Match: move from (j-1, i-1) to (j, i) while consuming one query character.
                if (j > 0 && i > 0) {
                    /*               double bestScore = NEG_INF;
                                   DPCell* bestPrev = nullptr;
                                   std::size_t prev_j = j - 1;
                                   std::size_t prev_i = i - 1;
                                   for (State s : {State::M, State::I, State::D}) {
                                       const DPCell& prevCell = dp(prev_j, prev_i, s);
                                       std::string key = (s == State::M) ? "MM" : (s == State::I) ? "IM" : "DM";
                                       double trans = NEG_INF;
                                       if (auto it = m_t[prev_j].find(key); it != m_t[prev_j].end())
                                           trans = it->second;
                                       double candidate = prevCell.score + trans;
                                       if (candidate > bestScore) {
                                           bestScore = candidate;
                                           bestPrev = const_cast<DPCell*>(&prevCell);
                                       }
                                   }*/

                    std::size_t prev_j = j - 1;
                    std::size_t prev_i = i - 1;
                    auto thing = getBestTrans(dp, prev_j, prev_i, State::M);
                    //                        assert(thing.score == bestScore);
                    //                        assert(thing.prev == bestPrev);

                    // Add emission cost for match from m_eM[j]
                    std::string letter(1, query[i - 1]);
                    double emission = NEG_INF;
                    if (auto it = m_eM[j].find(letter); it != m_eM[j].end())
                        emission = it->second;
                    //                    bestScore += emission;
                    thing.score += emission;

                    DPCell& cell = dp(j, i, State::M);
                    cell = thing;
                    //                    cell.score = thing.score;
                    //                    cell.prev = thing.prev;
                    //                    cell.state = State::M;
                    // Match: output the query letter in uppercase.
                    cell.alignedChar = std::toupper(query[i - 1]);
                }
            }
        }

        // --- Final transition ---
        /*        double finalScore = NEG_INF;
                DPCell* finalCell = nullptr;
                for (State s : {State::M, State::I, State::D}) {
                    const DPCell& cell = dp(K, L, s);
                    std::string key = (s == State::M) ? "MM" : (s == State::I) ? "IM" : "DM";
                    double trans = NEG_INF;
                    if (auto it = m_t[K].find(key); it != m_t[K].end())
                        trans = it->second;
                    double candidate = cell.score + trans;
                    if (candidate > finalScore) {
                        finalScore = candidate;
                        finalCell = const_cast<DPCell*>(&cell);
                    }
                }*/

        auto last = getBestTrans(dp, K, L, State::M);
        //        assert(last.score == finalScore);
        //        assert(last.prev == finalCell);

        if (last.score == NEG_INF || last.prev == nullptr)
            return {NEG_INF, "$"};

        // --- Backtracking ---
        // Now, follow the pointer chain (each cell stores its alignedChar)
        // to reconstruct the alignment string.
        std::string alignment;
        for (DPCell* cell = last.prev; cell->prev != nullptr; cell = cell->prev) {
            alignment.push_back(cell->alignedChar);
        }
        std::ranges::reverse(alignment);
        return {last.score, alignment};
    }
};

//
// Example usage:
//
int main() {
    HMM hmm;
    // For this example we assume three match states.
    // Set number of states.
    hmm.m_nStates = 4;

    // Set alphabet.
    hmm.m_alphabet = {"A", "C", "G", "T"};

    // Set insertion emission probabilities (eI) for states 0 to 4.
    hmm.m_eI = {
        {{"A", -1.386}, {"C", -1.386}, {"G", -1.386}, {"T", -1.386}},
        {{"A", -1.386}, {"C", -1.386}, {"G", -1.386}, {"T", -1.386}},
        {{"A", -1.386}, {"C", -1.386}, {"G", -1.386}, {"T", -1.386}},
        {{"A", -1.386}, {"C", -1.386}, {"G", -1.386}, {"T", -1.386}},
        {{"A", -1.386}, {"C", -1.386}, {"G", -1.386}, {"T", -1.386}}
    };

    // Set match emission probabilities (eM), with the first map left empty.
    hmm.m_eM = {
        {},
        {{"A", -0.693}, {"C", -1.204}, {"G", -2.303}, {"T", -2.303}},
        {{"A", -0.693}, {"C", -1.204}, {"G", -2.303}, {"T", -2.303}},
        {{"A", -0.693}, {"C", -1.204}, {"G", -2.303}, {"T", -2.303}},
        {{"A", -0.693}, {"C", -1.204}, {"G", -2.303}, {"T", -2.303}}
    };

    // Set transition probabilities (t) for rows 0 to 4.
    hmm.m_t = {
        {
            {"MM", -0.357}, {"MD", -1.897}, {"MI", -1.897}, {"IM", -0.223},
            {"II", -1.897}, {"ID", -2.996}, {"DM", -0.0}, {"DI", NEG_INF}, {"DD", NEG_INF}
        },
        {
            {"MM", -0.357}, {"MD", -1.897}, {"MI", -1.897}, {"IM", -0.223},
            {"II", -1.897}, {"ID", -2.996}, {"DM", -0.288}, {"DI", NEG_INF}, {"DD", -1.386}
        },
        {
            {"MM", -0.357}, {"MD", -1.897}, {"MI", -1.897}, {"IM", -0.223},
            {"II", -1.897}, {"ID", -2.996}, {"DM", -0.288}, {"DI", NEG_INF}, {"DD", -1.386}
        },
        {
            {"MM", -0.357}, {"MD", -1.897}, {"MI", -1.897}, {"IM", -0.223},
            {"II", -1.897}, {"ID", -2.996}, {"DM", -0.288}, {"DI", NEG_INF}, {"DD", -1.386}
        },
        {
            {"MM", -0.163}, {"MD", NEG_INF}, {"MI", -1.897}, {"IM", -0.163},
            {"II", -1.897}, {"ID", NEG_INF}, {"DM", -0.0}, {"DI", NEG_INF}, {"DD", NEG_INF}
        }
    };


    // Example query sequence.
    std::string query = "A";
    auto [score, alignment] = hmm.viterbi(query);
    std::cout << "Score: " << score << '\n';
    std::cout << "Alignment: " << alignment << '\n';
}