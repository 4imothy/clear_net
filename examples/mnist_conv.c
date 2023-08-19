#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"
#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"
#include <dirent.h>
#include <sys/stat.h>

const size_t img_height = 28;
const size_t img_width = 28;
const size_t num_pixels = img_height * img_width;
const size_t num_train_files = 60000;
const size_t num_test_files = 10000;
const size_t dim_output = 10;
const size_t nchannels = 1;

int get_data_from_dir(Matrix *data, Vector *targets, char *path, size_t num_files) {
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
            uint8_t *img_pixels = stbi_load(
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
                size_t row = j / img_width;
                size_t col = j % img_width;
                MAT_AT(data[count], row, col) = img_pixels[j] / 255.f;
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
    Matrix *train = CLEAR_NET_ALLOC(num_train_files * sizeof(Matrix));
    Vector *train_targets = CLEAR_NET_ALLOC(num_train_files * sizeof(Vector));
    for (size_t i = 0; i < num_train_files; ++i) {
        train[i] = cn_alloc_matrix(img_height, img_width);
        train_targets[i] = cn_alloc_vector(dim_output);
    }
    int res = get_data_from_dir(train, train_targets, train_path, num_train_files);
    if (res) {
       return 1;
    }

    /* char *test_path = "./datasets/mnist/test"; */
    /* Matrix *test = CLEAR_NET_ALLOC(num_test_files * sizeof(Matrix)); */
    /* Vector *test_targets = CLEAR_NET_ALLOC(num_test_files * sizeof(Vector)); */
    /* for (size_t i = 0; i < num_train_files; ++i) { */
    /*     test[i] = cn_alloc_matrix(img_width, img_height); */
    /*     test_targets[i] = cn_alloc_vector(dim_output); */
    /* } */

    /* res = get_data_from_dir(test, test_targets, test_path, num_test_files); */
    /* if (res != 0) { */
    /*     return 1; */
    /* } */

    // TODO make this a full convolutional net, and then fix the input randomization
    cn_default_hparams();
    Net net = cn_init_net();
    cn_alloc_convolutional_layer(&net, Valid, Sigmoid, nchannels, 3, img_height, img_width, 8, 8);
    cn_alloc_secondary_convolutional_layer(&net, Valid, LeakyReLU, 10, 3, 3);
    cn_alloc_global_pooling_layer(&net, Average);
    cn_alloc_secondary_dense_layer(&net, Tanh, 15);
    cn_alloc_secondary_dense_layer(&net, Sigmoid, 10);
    cn_randomize_net(net, -1, 1);

    Matrix **input_list = CLEAR_NET_ALLOC(num_train_files * sizeof(Matrix*));
    for (size_t i = 0; i < num_train_files; ++i) {
        input_list[i] = &train[i];
    }

    LaData *targets = CLEAR_NET_ALLOC(num_train_files * sizeof(LaData));
    for (size_t i = 0; i < num_train_files; ++i) {
        targets[i].type = Vec;
        targets[i].data.vec = train_targets[i];
    }

    // TODO this needs to randomize the inputs also
    // TODO do this first
    // cn_shuffle_conv_input(input_list, num_train_files);

    // Make sure these are the same
    // LaData vec = cn_predict_conv(&net, train);
    // _cn_print_vector(vec.data.vec, "vec out");
    // cn_learn_convolutional(&net, &train, targets, 1);

    size_t nepochs = 2000;
    size_t batch_size = 100;
    CLEAR_NET_ASSERT(num_train_files % batch_size == 0);
    cn_set_rate(0.05);
    // printf("Initial Cost: %f\n", cn_loss_conv(&net, input_list, targets, num_train_files));
    printf("Beginning Training\n");

    /* Matrix **batch_in = CLEAR_NET_ALLOC(batch_size * sizeof(Matrix*)); */
    /* LaData *batch_tar = CLEAR_NET_ALLOC(batch_size * sizeof(LaData)); */
    /* for (size_t i = 0; i < nepochs; ++i) { */
    /*     for (size_t batch_num = 0; batch_num < (num_train_files / batch_size); ++batch_num) { */
    /*         cn_get_batch_conv(batch_in, batch_tar, input_list, targets, batch_num, batch_size); */
    /*         printf("loss at batch: %zu is %f\n", batch_num, cn_learn_convolutional(&net, batch_in, batch_tar, batch_size)); */
    /*     } */
    /*     printf("Loss at epoch %zu: %f\n", i, cn_loss_conv(&net, input_list, targets, num_train_files)); */
    /* } */

    for (size_t i = 0; i < nepochs; ++i) {
        printf("Loss at epoch %zu is %f\n", i, cn_learn_convolutional(&net, input_list, targets, 10));
    }

    return 0;
}
