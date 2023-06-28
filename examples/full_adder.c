#include <stdio.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

#define BITS_PER_NUM 1

// a full adder with carry in and carry out
int main() {
  srand(0);
    // clang-format off
    float data[] = {
        // a  b  cin  sum  cout
        0,  0,  0,   0,   0,
        0,  0,  1,   1,   0,
        0,  1,  0,   1,   0,
        0,  1,  1,   0,   1,
        1,  0,  0,   1,   0,
        1,  0,  1,   0,   1,
        1,  1,  0,   0,   1,
        1,  1,  1,   1,   1,
    };
    // clang-format on
	
	// 2^3
	size_t num_combinations = 8;
	// a, b, cin
	size_t num_inputs = 3;
	// sum, cout
	size_t num_outputs = 2;
	Matrix input = {
	  .elements = data,
	  .nrows = num_combinations,
	  .ncols = num_inputs,
	  .stride = 5,
	};
	  Matrix output = {
	  .elements = &data[num_inputs],
	  .nrows = num_combinations,
	  .ncols = num_outputs,
	  .stride = 5,
	  };

	  size_t num_epochs = 20000;
   // size_t shape[] = {num_inputs, 3, 8, num_outputs};
	  size_t shape[] = {3, 3, 8, 2};
	  Net net = alloc_net(shape, ARR_LEN(shape));
	  net_rand(net, -1, 1);
	  for (size_t i = 0; i < num_epochs; ++i) {
		net_backprop(net, input, output);
		// TODO look at why this hangs sometimes probably with the g net that shouldn't exist
		//     if (i % (num_epochs / 5) == 0) {
  printf("Cost at %zu: %f\n", i, net_errorf(net, input, output));
  //        }
   	  }
	  return 0;
}  
