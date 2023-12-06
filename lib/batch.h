#pragma once

#include "neuralnettmp.h"

class Batch {
 public:
  Batch();

  void process(Region& r, size_t batch_size, NN& nn, const Matrix& t,
               float rate);
  bool isFinished() const;

 private:
  size_t begin;
  float cost;
  bool finished;
};
