#include "../include/nn.h"
#include <cstddef>
#include <cstdlib>

namespace nn {

Region region_alloc_alloc(size_t capacity_bytes) {
  return Region(capacity_bytes);
}

void *region_alloc(Region *r, size_t size) {
  if (r == nullptr) {
    return malloc(size);
  }
  return r->alloc(size);
}

void reset(Region *r) {
  if (r == nullptr) {
    return;
  }
  r->reset();
}

void save(Region *r) {
  if (r == nullptr) {
    return;
  }
  r->save();
}

void rewind(Region *r, size_t s) {
  if (r == nullptr) {
    return;
  }
  r->rewind(s);
}
} // namespace nn
