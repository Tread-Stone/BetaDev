#pragma once

#include <cstddef>
#include <memory>

class Region {
public:
  Region(size_t capacity_bytes);

  void *alloc(size_t size_bytes);
  void reset();
  size_t occupied_bytes() const;
  void save();
  void rewind(size_t s);

  // getters
  size_t get_capacity() const { return capacity; }
  size_t get_size() const { return size; }
  uintptr_t *get_elements() { return elements.get(); }
  const uintptr_t *get_elements() const { return elements.get(); }

private:
  size_t capacity;
  size_t size;
  std::unique_ptr<uintptr_t> elements;
};
