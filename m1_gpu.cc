#include "ece408net.h"
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace {

struct Args {
  int batch_size = 10000;
  bool competition_mode = false;
};

void print_usage(const char* program_name) {
  std::cerr << "Usage: " << program_name
            << " [batch_size] [--competition]" << std::endl;
}

bool parse_batch_size(const char* arg, int* batch_size) {
  errno = 0;
  char* end = NULL;
  long parsed = std::strtol(arg, &end, 10);
  if (errno != 0 || end == arg || *end != '\0' ||
      parsed <= 0 || parsed > INT_MAX) {
    return false;
  }
  *batch_size = static_cast<int>(parsed);
  return true;
}

bool parse_args(int argc, char* argv[], Args* args) {
  bool seen_batch_size = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--competition") == 0) {
      args->competition_mode = true;
      continue;
    }

    int parsed_batch_size = 0;
    if (!seen_batch_size && parse_batch_size(argv[i], &parsed_batch_size)) {
      args->batch_size = parsed_batch_size;
      seen_batch_size = true;
      continue;
    }

    return false;
  }

  return true;
}

}  // namespace

void inference_only(int batch_size) {

  std::cout<<"Loading fashion-mnist data...";
  MNIST dataset(get_data_dir());
  dataset.read_test_data(batch_size);
  std::cout<<"Done"<<std::endl;
  
  std::cout<<"Loading model...";
  Network dnn = createNetwork_GPU();
  std::cout<<"Done"<<std::endl;

  dnn.forward(dataset.test_data);
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout<<std::endl;
  std::cout<<"Test Accuracy: "<<acc<< std::endl;
  std::cout<<std::endl;
}

int main(int argc, char* argv[]) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    print_usage(argv[0]);
    return 1;
  }

  std::cout<<"Test batch size: "<<args.batch_size<<std::endl;
  if (args.competition_mode) {
    std::cout<<"Competition mode enabled"<<std::endl;
  }
  Conv_Custom::configure_benchmark(args.competition_mode, 5, 10);
  inference_only(args.batch_size);

  return 0;
}
