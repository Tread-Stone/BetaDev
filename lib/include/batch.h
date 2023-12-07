#pragma once

#include "nn.h"
#include "region.h"

class Batch {
public:
  Batch();

  void process(Region &r, size_t batch_size, nn::NN &nn, const Matrix &t,
               float rate);
  bool isFinished() const;

  // getters
  size_t get_begin() const { return begin; };
  float get_cost() const { return cost; };
  bool get_finished() const { return finished; };

private:
  size_t begin;
  float cost;
  bool finished;
};
