#include <chrono>
#include <string>
#include <iostream>
#include <torch/torch.h>
using namespace torch;

class Timer {
public:
    explicit Timer(const string& name) {
        start = std::chrono::steady_clock::now();
        this->name = name;
    }
    [[nodiscard]] std::chrono::duration<long double> Elapsed() const {
        return std::chrono::steady_clock::now() - start;
    }
    std::chrono::duration<long double> Reset() {
        decltype(start) exstart = start;
        std::chrono::duration<long double> elapsed = std::chrono::steady_clock::now() - exstart;
        start = std::chrono::steady_clock::now();
        return elapsed;
    }
    string name;
    decltype(std::chrono::steady_clock::now()) start;
};
int main() {
    /*
     *
    Tensor tensor = torch::rand({4, 4, 4});
    std::cout << &tensor << "?" << std::endl;
    tensor = tensor.to(torch::device(torch::kCUDA).dtype(torch::kFloat64)).requires_grad_(true);
    std::cout << &tensor << "?" << std::endl;   */

    Timer timer("timer1");
    struct Net : torch::nn::Module{
        Net(int64_t N, int64_t M)
            : linear(register_module("linear", torch::nn::Linear(N, M)))
        {}
        torch::nn::Linear linear;
        torch::Tensor forward(const torch::Tensor& X) {
            return torch::relu(linear->forward(X));
        }
    };
    Net model(10, 10);
    model.to(torch::kCUDA);
    std::cout << timer.Reset().count() << std::endl;

    struct Net2 : torch::nn::Module{
        explicit Net2(int64_t N = 100, int64_t M = 100)
                : linear(register_module("linear", torch::nn::Linear(N * 3, M * 3)))
        {}
        torch::nn::Linear linear;
        torch::Tensor forward(const torch::Tensor& X) {
            return torch::relu(linear->forward(X));
        }
    };
    Net2 model2[100];
    for (auto & i : model2)
        i.to(torch::kCUDA);
    std::cout << timer.Reset().count() << std::endl;
    for (const auto & i : model2)
        std::cout << i << std::endl << std::endl;
    return 0;
}
