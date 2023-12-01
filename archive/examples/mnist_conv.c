#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"
#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"
#include <dirent.h>
#include <sys/stat.h>

CLEAR_NET_DEFINE_HYPERPARAMETERS

const size_t img_height = 28;
const size_t img_width = 28;
const size_t num_pixels = img_height * img_width;
const size_t num_train_files = 60000;
const size_t num_test_files = 10000;
const size_t dim_output = 10;
const size_t nchannels = 1;

int get_data_from_dir(Matrix *data, Vector *targets, char *path,
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
    char *train_path = "../datasets/mnist/train";
    Matrix *train = CLEAR_NET_ALLOC(num_train_files * sizeof(Matrix));
    Vector *train_targets = CLEAR_NET_ALLOC(num_train_files * sizeof(Vector));
    for (size_t i = 0; i < num_train_files; ++i) {
        train[i] = cn_alloc_matrix(img_height, img_width);
        train_targets[i] = cn_alloc_vector(dim_output);
    }
    int res =
        get_data_from_dir(train, train_targets, train_path, num_train_files);
    if (res) {
        return 1;
    }

    Matrix **input_list = CLEAR_NET_ALLOC(num_train_files * sizeof(Matrix *));
    for (size_t i = 0; i < num_train_files; ++i) {
        input_list[i] = &train[i];
    }

    LAData *targets = CLEAR_NET_ALLOC(num_train_files * sizeof(LAData));
    for (size_t i = 0; i < num_train_files; ++i) {
        targets[i].type = Vec;
        targets[i].data.vec = train_targets[i];
    }

    char *test_path = "../datasets/mnist/test";
    Matrix *test = CLEAR_NET_ALLOC(num_test_files * sizeof(Matrix));
    Vector *test_targets = CLEAR_NET_ALLOC(num_test_files * sizeof(Vector));
    for (size_t i = 0; i < num_test_files; ++i) {
        test[i] = cn_alloc_matrix(img_width, img_height);
        test_targets[i] = cn_alloc_vector(dim_output);
    }
    res = get_data_from_dir(test, test_targets, test_path, num_test_files);
    if (res != 0) {
        return 1;
    }

    Matrix **test_list = CLEAR_NET_ALLOC(num_test_files * sizeof(Matrix *));
    LAData *la_test_targets = CLEAR_NET_ALLOC(num_test_files * sizeof(LAData));
    for (size_t i = 0; i < num_test_files; ++i) {
        test_list[i] = &test[i];
        la_test_targets[i].type = Vec;
        la_test_targets[i].data.vec = test_targets[i];
    }

    cn_shuffle_conv_input(&input_list, &targets, num_train_files);

    cn_default_hparams();
    // cn_with_momentum(0.9);
    Net net = cn_alloc_conv_net(img_height, img_width, nchannels);
    cn_alloc_conv_layer(&net, Valid, Sigmoid, 3, 9, 9);
    cn_alloc_conv_layer(&net, Valid, Sigmoid, 5, 5, 5);
    cn_alloc_pooling_layer(&net, Average, 4, 4);
    cn_alloc_conv_layer(&net, Valid, Sigmoid, 10, 3, 3);
    cn_alloc_global_pooling_layer(&net, Max);
    cn_randomize_net(&net, -1, 1);

    size_t nepochs = 2000;
    size_t batch_size = 32;
    CLEAR_NET_ASSERT(num_train_files % batch_size == 0);

    cn_set_rate(0.01);
    printf("Initial Cost: %f\n",
           cn_loss_conv(&net, input_list, targets, num_train_files));
    printf("Beginning Training\n");

    Matrix **batch_in = CLEAR_NET_ALLOC(batch_size * sizeof(Matrix *));
    LAData *batch_tar = CLEAR_NET_ALLOC(batch_size * sizeof(LAData));
    size_t nbatches = num_train_files / batch_size;
    float loss;
    for (size_t i = 0; i < nepochs; ++i) {
        for (size_t batch_num = 0; batch_num < nbatches; ++batch_num) {
            cn_get_batch_conv(batch_in, batch_tar, input_list, targets,
                              batch_num, batch_size);
            printf("Loss at batch: %zu is %f\n", batch_num,
                   cn_learn_conv(&net, batch_in, batch_tar, batch_size));
        }
        loss = cn_loss_conv(&net, input_list, targets, num_train_files);
        printf("Loss at epoch %zu: %f\n", i, loss);
        if (loss < 0.25) {
            break;
        }
    }

    cn_print_conv_results(net, test_list, la_test_targets, num_test_files);

    char *file = "model";
    printf("Loss on validation: %f\n",
           cn_loss_conv(&net, test_list, la_test_targets, num_test_files));
    cn_save_net_to_file(net, file);
    cn_dealloc_net(&net);
    net = cn_alloc_net_from_file(file);
    printf("Loss on validation after loading save: %f\n",
           cn_loss_conv(&net, test_list, la_test_targets, num_test_files));

    return 0;
}
