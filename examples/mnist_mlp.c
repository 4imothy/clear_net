#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

#include <dirent.h>
#include <sys/stat.h>

const size_t img_height = 28;
const size_t img_width = 28;
const size_t num_pixels = img_height * img_width;
const size_t num_train_files = 60000;
const size_t num_test_files = 10000;
const size_t dim_output = 10;

float fix_output(float x) { return x; }

int get_data_from_dir(Matrix *data, char *path, int num_files) {
    DIR *directory = opendir(path);
    if (directory == NULL) {
        printf("Error: Failed to open %s.\n", path);
        return 1;
    }
    struct dirent *entry;
    int count = 0;

    while ((entry = readdir(directory)) != NULL) {
        // Skip dotfiles
        if (entry->d_name[0] == '.')
            continue;

        // Construct the file path
        char file_path[PATH_MAX];
        snprintf(file_path, PATH_MAX, "%s/%s", path, entry->d_name);

        // Check if the entry is a regular file
        struct stat file_stat;
        if (stat(file_path, &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
            int img_width, img_height, img_comp;
            uint8_t *img_pixels = (uint8_t *)stbi_load(
                file_path, &img_width, &img_height, &img_comp, 0);

            if (img_pixels == NULL) {
                fprintf(
                    stderr,
                    "ERROR: could not read %s\n Did you download the data? The "
                    "binary begins its search at the directory you call it.\n",
                    file_path);
                return 1;
            }
            if (img_comp != 1) {
                fprintf(stderr, "ERROR: %s improperly formatted", file_path);
                return 1;
            }
            for (int j = 0; j < img_width * img_height; ++j) {
                MAT_AT(*data, count, j) = img_pixels[j] / 255.f;
            }
            // the python script set it up so the first character is the label
            size_t label = (entry->d_name[0] - '0');
            MAT_AT(*data, count, num_pixels + label) = 1;
            count++;
        }
    }
    closedir(directory);
    CLEAR_NET_ASSERT(count == num_files);

    return 0;
}

int main(void) {
    srand(0);

    char *train_path = "./datasets/mnist/train";
    Matrix train = cn_alloc_matrix(num_train_files, num_pixels + dim_output);
    int res = get_data_from_dir(&train, train_path, num_train_files);
    if (res) {
        return 1;
    }
    // randomize for stochastic gradient descent
    cn_shuffle_matrix_rows(train);
    Matrix train_input = cn_form_matrix(num_train_files, num_pixels,
                                        train.ncols, &MAT_AT(train, 0, 0));
    Matrix train_output =
        cn_form_matrix(num_train_files, dim_output, train.ncols,
                       &MAT_AT(train, 0, num_pixels));

    char *test_path = "./datasets/mnist/test";
    Matrix test = cn_alloc_matrix(num_test_files, num_pixels + dim_output);
    res = get_data_from_dir(&test, test_path, num_test_files);
    if (res != 0) {
        return 1;
    }
    Matrix test_input = cn_form_matrix(num_test_files, num_pixels, test.ncols,
                                       &MAT_AT(test, 0, 0));
    Matrix test_output = cn_form_matrix(num_test_files, dim_output, test.ncols,
                                        &MAT_AT(test, 0, num_pixels));

    cn_with_momentum(0.9);
    Net net = cn_init_net();
    cn_alloc_dense_layer(&net, num_pixels, 16, Sigmoid);
    cn_alloc_dense_layer(&net, 16, 16, Sigmoid);
    cn_alloc_dense_layer(&net, 16, dim_output, Sigmoid);
    cn_randomize_net(net, -1, 1);
    size_t num_epochs = 20000;
    float error;
    float error_break = 0.10;
    // Use stochastic gradient descent
    Matrix batch_input;
    Matrix batch_output;
    size_t batch_size = 100;
    CLEAR_NET_ASSERT(num_train_files % batch_size == 0);
    printf("Beginning Training\n");
    printf("Initial Cost: %f\n", cn_loss_mlp(net, train_input, train_output));
    for (size_t i = 0; i < num_epochs; ++i) {
        for (size_t batch_num = 0; batch_num < (num_train_files / batch_size);
             ++batch_num) {
            cn_get_batch(&batch_input, &batch_output, train_input, train_output,
                         batch_num, batch_size);
            cn_learn_mlp(&net, batch_input, batch_output);
        }
        error = cn_loss_mlp(net, train_input, train_output);
        printf("Cost after epoch %zu: %f\n", i, error);
        if (error < error_break) {
            printf("Less than: %f error after epoch %zu\n", error_break, i);
            break;
        }
    }

    printf("Final Error on training set: %f\n",
           cn_loss_mlp(net, train_input, train_output));
    char *file = "model";
    cn_save_net_to_file(net, file);
    cn_dealloc_net(&net);
    net = cn_alloc_net_from_file(file);
    printf("On training\n");
    cn_print_target_output_pairs_mlp(net, train_input, train_output);
    printf("On testing\n");
    cn_print_target_output_pairs_mlp(net, test_input, test_output);
    cn_dealloc_net(&net);
    cn_dealloc_matrix(&train);
    cn_dealloc_matrix(&test);
}
