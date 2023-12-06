#include "neuralnettmp.h"

namespace neuralnet {

/**
 * Actual alloc
 */
Region region_alloc_alloc(size_t capacity_bytes) {
  Region r = {0};

  size_t elements_size = sizeof(*r.elements);
  size_t capacity = (capacity_bytes + elements_size - 1) / elements_size;

  void *elements = NN_MALLOC(capacity * elements_size);
  NN_ASSERT(elements != NULL);
  r.elements = elements;
  r.capacity = capacity;

  return r;
}

void *region_alloc(Region *r, size_t size) {
  if (r == NULL) return NN_MALLOC(size);
  size_t word_size = sizeof(*r->elements);
  size_t count = (size + word_size - 1) / word_size;

  NN_ASSERT(r->size + count <= r->capacity);
  if (r->size + count > r->capacity) return NULL;
  void *result = &r->elements[r->size];
  r->size += count;
  return result;
}

}  // namespace neuralnet
