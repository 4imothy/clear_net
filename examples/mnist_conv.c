#define STB_IMAGE_IMPLEMENTATION
#include "../lib/clear_net.h"
#include "./external/stb_image.h"
#include <dirent.h>
#include <sys/stat.h>

#define data cn.data

const size_t img_height = 28;
const size_t img_width = 28;
const size_t num_pixels = img_height * img_width;
const size_t num_train_files = 60000;
const size_t num_test_files = 10000;
const size_t dim_output = 10;
const size_t nchannels = 1;

int get_data_from_dir(Matrix *train, Vector *targets, char *path,
                      size_t num_files) {
    DIR *directory = opendir(path);
    if (directory == NULL) {
        printf("Error: Failed to open %s.\n", path);
        return 1;
    }
    struct dirent *entry;
    size_t count = 0;

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
            uint8_t *img_pixels =
                stbi_load(file_path, &img_width, &img_height, &img_comp, 0);
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
                size_t row = j / img_width;
                size_t col = j % img_width;
                MAT_AT(train[count], row, col) = img_pixels[j] / 255.f;
            }
            // the python script set it up so the first character is the label
            size_t label = (entry->d_name[0] - '0');
            VEC_AT(targets[count], label) = 1;
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
    Matrix *train_ins =
        data.allocMatrices(num_train_files, img_height, img_width);
    Vector *train_tars = data.allocVectors(num_train_files, dim_output);
    int res =
        get_data_from_dir(train_ins, train_tars, train_path, num_train_files);
    if (res) {
        return 1;
    }

    CNData *input = data.allocDataFromMatrices(train_ins, num_train_files);
    CNData *target = data.allocDataFromVectors(train_tars, num_train_files);

    char *test_path = "./datasets/mnist/test";
    Matrix *test_ins =
        data.allocMatrices(num_test_files, img_height, img_width);
    Vector *test_tars = data.allocVectors(num_test_files, dim_output);
    res = get_data_from_dir(test_ins, test_tars, test_path, num_test_files);
    if (res != 0) {
        return 1;
    }
    CNData *test_input = data.allocDataFromMatrices(test_ins, num_test_files);
    CNData *test_target = data.allocDataFromVectors(test_tars, num_test_files);

    data.shuffleDatas(input, target);

    HParams *hp = cn.allocDefaultHParams();

    cn.setRate(hp, 0.01);
    // cn_with_momentum(0.9);
    Net *net = cn.allocConvNet(hp, img_height, img_width, nchannels);
    cn.allocConvLayer(net, SIGMOID, VALID, 3, 9, 9);
    cn.allocConvLayer(net, SIGMOID, VALID, 5, 5, 5);
    cn.allocPoolingLayer(net, AVERAGE, 4, 4);
    cn.allocConvLayer(net, SIGMOID, VALID, 10, 3, 3);
    cn.allocGlobalPoolingLayer(net, MAX);
    cn.randomizeNet(net, -1, 1);

    size_t nepochs = 2000;
    size_t batch_size = 32;
    CLEAR_NET_ASSERT(num_train_files % batch_size == 0);
    printf("Initial Cost: %f\n", cn.lossConv(net, input, target));
    printf("Beginning Training\n");

    CNData *batch_in = data.allocEmptyData();
    CNData *batch_tar = data.allocEmptyData();
    size_t nbatches = num_train_files / batch_size;
    float loss;
    for (size_t i = 0; i < nepochs; ++i) {
        for (size_t batch_num = 0; batch_num < nbatches; ++batch_num) {
            data.setBatch(input, target, batch_num, batch_size, batch_in,
                          batch_tar);
            printf("Loss at batch: %zu is %f\n", batch_num,
                   cn.lossConv(net, batch_in, batch_tar));
            cn.backprop(net);
        }
        loss = cn.lossConv(net, input, target);
        printf("Loss at epoch %zu: %f\n", i, loss);
        if (loss < 0.25) {
            break;
        }
    }

    char *file = "model";
    printf("Loss on validation: %f\n",
           cn.lossConv(net, test_input, test_target));
    cn.saveNet(net, file);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file);
    printf("Loss on validation after loading save: %f\n",
           cn.lossConv(net, test_input, test_target));

    return 0;
}
