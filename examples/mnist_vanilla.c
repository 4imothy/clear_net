#define STB_IMAGE_IMPLEMENTATION
#include "../lib/clear_net.h"
#include "./external/stb_image.h"

#include <dirent.h>
#include <sys/stat.h>

#define data cn.data

const ulong img_height = 28;
const ulong img_width = 28;
const ulong num_pixels = img_height * img_width;
const ulong num_train_files = 60000;
const ulong num_test_files = 10000;
const ulong dim_output = 10;

int get_data_from_dir(Vector *train, Vector *targets, char *path,
                      ulong num_files) {
    DIR *directory = opendir(path);
    if (directory == NULL) {
        printf("Error: Failed to open %s.\n", path);
        return 1;
    }
    struct dirent *entry;
    ulong count = 0;

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
                VEC_AT(train[count], j) = img_pixels[j] / 255.f;
            }
            // the python script set it up so the first character is the label
            ulong label = (entry->d_name[0] - '0');
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
    Vector *vinputs = data.allocVectors(num_train_files, num_pixels);
    Vector *vtargets = data.allocVectors(num_train_files, dim_output);

    int res = get_data_from_dir(vinputs, vtargets, train_path, num_train_files);
    if (res) {
        return 1;
    }

    CNData *inputs = data.allocDataFromVectors(vinputs, num_train_files);
    CNData *targets = data.allocDataFromVectors(vtargets, num_train_files);

    // randomize for stochastic gradient descent
    data.shuffleDatas(inputs, targets);

    char *test_path = "./datasets/mnist/test";
    Vector *vtest_in = data.allocVectors(num_test_files, num_pixels);
    Vector *vtest_targets = data.allocVectors(num_test_files, dim_output);

    res = get_data_from_dir(vtest_in, vtest_targets, test_path, num_test_files);
    if (res != 0) {
        return 1;
    }

    CNData *test_ins = data.allocDataFromVectors(vtest_in, num_train_files);
    CNData *test_tars =
        data.allocDataFromVectors(vtest_targets, num_train_files);

    HParams *hp = cn.allocDefaultHParams();
    cn.setRate(hp, 0.005);
    cn.withMomentum(hp, 0.9);
    Net *net = cn.allocVanillaNet(hp, num_pixels);
    cn.allocDenseLayer(net, SIGMOID, 16);
    cn.allocDenseLayer(net, SIGMOID, 16);
    cn.allocDenseLayer(net, SIGMOID, dim_output);
    cn.randomizeNet(net, -1, 1);
    ulong num_epochs = 1;
    scalar error;
    scalar error_break = 0.10;
    ulong batch_size = 100;
    CLEAR_NET_ASSERT(num_train_files % batch_size == 0);
    printf("Initial Cost: %f\n", cn.lossVanilla(net, inputs, targets));
    printf("Beginning Training\n");
    // for SGD
    CNData *batch_ins = data.allocEmptyData();
    CNData *batch_tars = data.allocEmptyData();
    for (ulong i = 0; i < num_epochs; ++i) {
        for (ulong batch_num = 0; batch_num < (num_train_files / batch_size);
             ++batch_num) {
            data.setBatch(inputs, targets, batch_num, batch_size, batch_ins,
                          batch_tars);
            cn.lossVanilla(net, batch_ins, batch_tars);
            cn.backprop(net);
        }
        error = cn.lossVanilla(net, inputs, targets);
        printf("Cost after epoch %zu: %f\n", i, error);
        if (error < error_break) {
            printf("Less than: %f error after epoch %zu\n", error_break, i);
            break;
        }
    }

    printf("Final Error on training set: %f\n",
           cn.lossVanilla(net, inputs, targets));
    char *file = "model";
    cn.saveNet(net, file);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file);
    printf("Testing Predictions\n");
    cn.printVanillaPredictions(net, test_ins, test_tars);
    cn.deallocNet(net);
}
