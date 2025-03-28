#ifndef PYBNESIAN_UTIL_PROGRESS_HPP
#define PYBNESIAN_UTIL_PROGRESS_HPP

#include <indicators/indicators.hpp>
// #include <sstream>
// #include <boost/format.hpp>  // sudo apt install libboost-all-dev
#include <iostream>
// enum class log_level_t { LOG_NOTHING, LOG_CRITICAL, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG };

namespace util {

// auto GLOBAL_LEVEL = log_level_t::LOG_INFO;
class formatted_log_t {
public:
    formatted_log_t(int verbose_level, std::string msg) : verbose_level(verbose_level), msg(msg) {}
    ~formatted_log_t() {
        // GLOBAL_LEVEL is a global variable and could be changed at runtime
        // Any customization could be here
        // if (level <= GLOBAL_LEVEL)
        //     std::wcout << static_cast<int>(level) << L" " << fmt.str()
        //                << std::endl;  // Convert level to a string before printing
        if (verbose_level > 0) {
            std::cout << msg << std::endl;
        }
    }
    // template <typename T>
    // formatted_log_t& operator%(T value) {
    //     fmt % value;
    //     return *this;
    // }
    // formatted_log_t log(int verbose_level, const char* msg) { return formatted_log_t(verbose_level, msg); }

protected:
    int verbose_level;
    std::string msg;
};
// Helper function. Class formatted_log_t will not be used directly.
// template <bool verbose_level>
// formatted_log_t log(const char* msg) {
//     return formatted_log_t(verbose_level, msg);
// }

class BaseIndeterminateSpinner {
public:
    virtual ~BaseIndeterminateSpinner() {}
    virtual void update_status(const std::string& s) = 0;
    virtual void update_status(std::string&& s) = 0;
    virtual void update_status() = 0;
    virtual void mark_as_completed(const std::string& s) = 0;
    virtual void mark_as_completed(std::string&& s) = 0;
    virtual void mark_as_completed() = 0;
};

class VoidProgressSpinner : public BaseIndeterminateSpinner {
public:
    VoidProgressSpinner() = default;

    void update_status(const std::string&) override {}
    void update_status(std::string&&) override {}
    void update_status() override {}
    void mark_as_completed(const std::string&) override {}
    void mark_as_completed(std::string&&) override {}
    void mark_as_completed() override {}
};

class IndeterminateSpinner : public BaseIndeterminateSpinner {
public:
    IndeterminateSpinner()
        : m_spinner(indicators::option::PostfixText{"Checking dataset..."},
                    indicators::option::MaxPostfixTextLen{0},
                    indicators::option::ShowPercentage{false},
                    indicators::option::ShowElapsedTime{true},
                    indicators::option::SpinnerStates{std::vector<std::string>{"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"}},
                    indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}) {}

    template <typename... Args>
    IndeterminateSpinner(Args&&... args) : m_spinner(args...) {}

    void update_status(const std::string& s) override {
        m_spinner.set_option(indicators::option::PostfixText{s});
        update_status();
    }
    void update_status(std::string&& s) override { update_status(s); }

    void update_status() override { m_spinner.set_progress(1); }

    void mark_as_completed(const std::string& s) override {
        m_spinner.set_option(indicators::option::PrefixText{"✔"});
        m_spinner.set_option(indicators::option::ShowSpinner{false});
        m_spinner.set_option(indicators::option::PostfixText{s});
        m_spinner.set_option(indicators::option::ForegroundColor{indicators::Color::green});
        mark_as_completed();
    }

    void mark_as_completed(std::string&& s) override { mark_as_completed(s); }

    void mark_as_completed() override { m_spinner.mark_as_completed(); }

private:
    indicators::ProgressSpinner m_spinner;
};

/**
 * @brief Creates a spinner based on the verbose level.
 *
 * @tparam Args Arguments to pass to the spinner.
 * @param verbose_level 0: no spinner, 1: indeterminate spinner
 * @param additional_args Additional arguments to pass to the spinner.
 * @return std::unique_ptr<BaseIndeterminateSpinner> Pointer to the spinner.
 */
template <typename... Args>
std::unique_ptr<BaseIndeterminateSpinner> indeterminate_spinner(int verbose_level, Args&&... additional_args) {
    switch (verbose_level) {
        case 0:
            return std::make_unique<VoidProgressSpinner>();
        case 1: {
            if constexpr (sizeof...(Args) == 0)
                return std::make_unique<IndeterminateSpinner>();
            else
                return std::make_unique<IndeterminateSpinner>(additional_args...);
        }
        default:
            throw std::invalid_argument("Wrong verbose level. Allowed values are 0 and 1.");
    }
}

class BaseProgressBar {
public:
    virtual ~BaseProgressBar() {}
    virtual void set_text(const std::string& s) = 0;
    virtual void set_text(std::string&& s) = 0;
    virtual void set_max_progress(int max_progress) = 0;
    virtual void add_progress(int progress) = 0;
    virtual void set_progress(int progress) = 0;
    virtual void tick() = 0;
    virtual void mark_as_completed(const std::string& s) = 0;
    virtual void mark_as_completed(std::string&& s) = 0;
    virtual void mark_as_completed() = 0;
    virtual void clean_terminal() = 0;
    virtual int verbose_level() = 0;
};

class VoidProgressBar : public BaseProgressBar {
public:
    VoidProgressBar() = default;
    void set_text(const std::string&) override {}
    void set_text(std::string&&) override {}
    void set_max_progress(int) override {}
    void add_progress(int) override {}
    void set_progress(int) override {}
    void tick() override {}
    void mark_as_completed(const std::string&) override {}
    void mark_as_completed(std::string&&) override {}
    void mark_as_completed() override {}
    void clean_terminal() override {}
    int verbose_level() override { return 0; }
};

class ProgressBar : public BaseProgressBar {
public:
    ProgressBar()
        : m_bar(indicators::option::BarWidth{40},
                indicators::option::Start{"["},
                indicators::option::End{"]"},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ForegroundColor{indicators::Color::white}) {}

    ProgressBar(int max_progress)
        : m_bar(indicators::option::BarWidth{40},
                indicators::option::Start{"["},
                indicators::option::End{"]"},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ForegroundColor{indicators::Color::white},
                indicators::option::MaxProgress{max_progress}) {}

    template <typename... Args>
    ProgressBar(Args&&... args) : m_bar(args...) {}

    void set_text(const std::string& s) override { m_bar.set_option(indicators::option::PostfixText{s}); }

    void set_text(std::string&& s) override { set_text(s); }

    void set_max_progress(int max_progress) override {
        m_bar.set_option(indicators::option::MaxProgress{max_progress});
    }

    void add_progress(int progress) override { m_bar.set_progress(m_bar.current() + progress); }

    void set_progress(int progress) override { m_bar.set_progress(progress); }

    void tick() override { m_bar.tick(); }

    void mark_as_completed(const std::string& s) override {
        m_bar.set_option(indicators::option::PrefixText{"✔  "});
        m_bar.set_option(indicators::option::PostfixText{s});
        m_bar.set_option(indicators::option::ForegroundColor{indicators::Color::green});
        mark_as_completed();
    }

    void mark_as_completed(std::string&& s) override { mark_as_completed(s); }

    void mark_as_completed() override { m_bar.mark_as_completed(); }

    void clean_terminal() override { std::cout << std::string(indicators::terminal_width(), ' ') << "\r"; }

    int verbose_level() override { return 1; };

private:
    indicators::BlockProgressBar m_bar;
};

template <typename... Args>
std::unique_ptr<BaseProgressBar> progress_bar(int verbose_level, Args&&... additional_args) {
    switch (verbose_level) {
        case 0:
            return std::make_unique<VoidProgressBar>();
        case 1: {
            if constexpr (sizeof...(Args) == 0)
                return std::make_unique<ProgressBar>();
            else
                return std::make_unique<ProgressBar>(additional_args...);
        }
        default:
            throw std::invalid_argument("Wrong verbose level. Allowed values are 0 and 1.");
    }
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_PROGRESS_HPP