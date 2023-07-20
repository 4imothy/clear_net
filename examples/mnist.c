#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

#include <dirent.h>
#include <sys/stat.h>

const size_t img_height = 28;
const size_t img_width = 28;
const size_t num_pixels = img_height * img_width;
const size_t num_train_files = 60000;
const size_t num_test_files = 10000;

float fix_output(float x) { return 10 * x; }

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
                MAT_GET(*data, count, j) = img_pixels[j] / 255.f;
            }
            // the python script set it up so the first character is the label
            MAT_GET(*data, count, num_pixels) = (entry->d_name[0] - '0') / 10.f;
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
    Matrix train = alloc_mat(num_train_files, num_pixels + 1);
    int res = get_data_from_dir(&train, train_path, num_train_files);
    if (res != 0) {
        return 1;
    }
    // randomize for stochastic gradient descent
    mat_randomize_rows(train);
    Matrix train_input = mat_form(num_train_files, num_pixels, train.ncols,
                                  &MAT_GET(train, 0, 0));
    Matrix train_output = mat_form(num_train_files, 1, train.ncols,
                                   &MAT_GET(train, 0, num_pixels));

    char *test_path = "./datasets/mnist/test";
    Matrix test = alloc_mat(num_train_files, num_pixels + 1);
    res = get_data_from_dir(&test, test_path, num_test_files);
    if (res != 0) {
        return 1;
    }
    Matrix test_input =
        mat_form(num_test_files, num_pixels, test.ncols, &MAT_GET(test, 0, 0));
    Matrix test_output =
        mat_form(num_test_files, 1, test.ncols, &MAT_GET(test, 0, num_pixels));

    // size_t shape[] = {num_pixels, 20, 16, 1};
    size_t shape[] = {num_pixels, 16, 16, 1};
    size_t num_layers = ARR_LEN(shape);
    Net net = alloc_net(shape, num_layers);
    net_rand(net, -1, 1);
    size_t num_epochs = 20000;
    float error;
    float error_break = 0.01;
    // Use stochastic gradient descent
    Matrix batch_input;
    Matrix batch_output;
    size_t batch_size = 10;
    CLEAR_NET_ASSERT(num_train_files % batch_size == 0);
    printf("Beginning Training\n");
    printf("Initial Cost: %f\n", net_errorf(net, train_input, train_output));
    for (size_t i = 0; i < num_epochs; ++i) {
        for (size_t batch_num = 0; batch_num < (num_train_files / batch_size);
             ++batch_num) {
            net_get_batch(&batch_input, &batch_output, train_input,
                          train_output, batch_num, batch_size);
            net_backprop(net, batch_input, batch_output);
        }
        error = net_errorf(net, train_input, train_output);
        printf("Cost after epoch %zu: %f\n", i, error);
        if (error < error_break) {
            printf("Less than: %f error after epoch %zu\n", error_break, i);
            break;
        }
    }

    printf("Final Error on training set: %f\n",
           net_errorf(net, train_input, train_output));
    char *file = "mnist_model";
    net_save_to_file(file, net);
    dealloc_net(&net);
    net = alloc_net_from_file(file);
    printf("After Loading, Error on training set: %f\n",
           net_errorf(net, train_input, train_output));
    printf("After Loading, Error on testing set: %f\n",
           net_errorf(net, test_input, test_output));
    printf("Actual | Prediction\n");
    for (size_t i = 0; i < num_test_files; ++i) {
        Matrix in = mat_row(test_input, i);
        mat_copy(NET_INPUT(net), in);
        net_forward(net);
        printf("%f | ", MAT_GET(test_output, i, 0));
        printf("%f\n", MAT_GET(NET_OUTPUT(net), 0, 0));
    }
    dealloc_mat(&train);
    dealloc_mat(&test);
}
