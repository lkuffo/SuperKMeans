#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace skmeans {

class Stopwatch {
  public:
    Stopwatch(std::string stopwatch_name) : stopwatch_name(stopwatch_name) {}

    using Clock = std::chrono::steady_clock;

    void start(std::string name) {
        always_assert(
            !m_activeName.has_value(), "Stopwatch::start called while another name is active"
        );

        m_activeName = name;
        m_startTime = Clock::now();
    }

    void stop(std::string name) {
        always_assert(m_activeName.has_value(), "Stopwatch::stop called with nothing active");

        always_assert(name == *m_activeName, "Stopwatch::stop name does not match active name");

        const auto endTime = Clock::now();
        const auto elapsed = endTime - m_startTime;

        m_accumulated[name] += elapsed;
        m_total += elapsed;

        m_activeName.reset();
    }

    void print() const {
        struct Row {
            std::string name;
            double ms;
            double pct;
        };

        std::vector<Row> rows;
        rows.reserve(m_accumulated.size());

        const double totalSec = std::chrono::duration<double>(m_total).count();

        for (const auto& [name, duration] : m_accumulated) {
            const double sec = std::chrono::duration<double>(duration).count();

            rows.push_back({name, sec * 1000.0, totalSec > 0.0 ? (sec / totalSec) * 100.0 : 0.0});
        }

        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ms > b.ms; });

        std::cout << "\nStopwatch " << stopwatch_name << " results:\n";

        std::cout << std::left << std::setw(12) << "Name" << std::right << std::setw(14)
                  << "Time (ms)" << std::setw(12) << "Percent\n";

        std::cout << std::string(38, '-') << '\n';

        std::cout << std::fixed << std::setprecision(3);

        for (const auto& r : rows) {
            std::cout << std::left << std::setw(12) << r.name << std::right << std::setw(14) << r.ms
                      << std::setw(11) << r.pct << "%\n";
        }

        std::cout << std::string(38, '-') << '\n';

        const double totalMs = std::chrono::duration<double, std::milli>(m_total).count();

        std::cout << std::left << std::setw(12) << "TOTAL" << std::right << std::setw(14) << totalMs
                  << std::setw(12) << "100.000%\n";
    }

  private:
    // Non-debug assertion: always active.
    static void always_assert(bool condition, const char* message) {
        if (!condition) {
            std::cerr << "Stopwatch assertion failed: " << message << '\n';
            std::terminate();
        }
    }

  private:
    std::string stopwatch_name;
    std::unordered_map<std::string, Clock::duration> m_accumulated;

    Clock::duration m_total{};

    std::optional<std::string> m_activeName;
    Clock::time_point m_startTime{};
};

} // namespace skmeans
