#include "ece408net.h"

int main(int argc, char* argv[]) {
  int image_idx = 0;
  if (argc >= 2)
    image_idx = atoi(argv[1]);

  int batch_size = std::max(image_idx + 1, 100);

  std::cout << "Loading fashion-mnist data...";
  MNIST dataset(get_data_dir());
  dataset.read_test_data(batch_size);
  std::cout << "Done" << std::endl;

  std::cout << "Loading model...";
  Network dnn = createNetwork_GPU();
  std::cout << "Done" << std::endl;

  std::string output_dir = "feature_maps";
  std::cout << "Dumping feature maps for image #" << image_idx
            << " to " << output_dir << "/" << std::endl;
  dnn.dump_feature_maps(dataset.test_data, dataset.test_labels,
                        output_dir, image_idx);
  std::cout << "Done. Run: python3 visualize.py " << output_dir << "/" << std::endl;
  return 0;
}
