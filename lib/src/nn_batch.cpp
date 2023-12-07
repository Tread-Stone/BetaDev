#include "../include/nn.h"

namespace neuralnet {

void batch_process(Region *r, Batch *b, size_t batch_size, nn::NN nn, Matrix t,
                   float rate) {
  if (b->finished) {
    b->finished = false;
    b->begin = 0;
    b->cost = 0;
  }

  size_t size = batch_size;
  if (b->begin + batch_size >= t.get_rows()) {
    size = t.get_rows() - b->begin;
  }

  // TODO: introduce similar to row_slice operation but for Mat that will give
  // you subsequence of rows
  Matrix batch_t = {
      .rows = size,
      .cols = t.get_cols(),
      .elements = &MAT_AT(t, b->begin, 0),
  };

  NN g = backprop(r, nn, batch_t);
  learn(nn, g, rate);
  b->cost += cost(nn, batch_t);
  b->begin += batch_size;

  if (b->begin >= t.get_rows()) {
    size_t batch_count = (t.get_rows() + batch_size - 1) / batch_size;
    b->cost /= batch_count;
    b->finished = true;
  }
}

} // namespace neuralnet
