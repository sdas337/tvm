/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_DATA_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_DATA_H_
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

#include "pipeline_struct.h"
#ifdef __cplusplus

#define read_barrier() std::atomic_thread_fence(std::memory_order_acquire)

template <typename SLOT_TYPE = SLOT>
squeue<SLOT_TYPE>* createQueue(squeue<SLOT_TYPE>* q, size_t size) {
  squeue<SLOT_TYPE>* rq = new squeue<SLOT_TYPE>();
  return rq;
}

template <typename SLOT_TYPE = SLOT>
void deleteQueue(squeue<SLOT_TYPE>* q) {
  free(q);
}

template <typename SLOT_TYPE = SLOT>
inline bool full(squeue<SLOT_TYPE>* q) {
  return ((q->tail + 1) % q->len) == q->head;
}

template <typename SLOT_TYPE = SLOT>
inline bool empty(squeue<SLOT_TYPE>* q) {
  return q->head == q->tail;
}

template <typename SLOT_TYPE = SLOT, typename VARIABLE_TYPE = SLOT>
void q_push(squeue<SLOT_TYPE>* q, const VARIABLE_TYPE& s) {
  while (full(q)) {
  }
  q->q[q->tail] = s;
  read_barrier();
  q->tail = (q->tail + 1) % q->len;
}

template <typename SLOT_TYPE = SLOT, typename VARIABLE_TYPE = SLOT>
inline bool q_poll_top_half(squeue<SLOT_TYPE>* q, VARIABLE_TYPE* s) {
  if (empty(q)) return false;
  *s = q->q[q->head];
  return true;
}

template <typename SLOT_TYPE = SLOT>
inline bool q_poll_bottom_half(squeue<SLOT_TYPE>* q) {
  q->head = (q->head + 1) % q->len;
  return true;
}

template <typename SLOT_TYPE = SLOT, typename VARIABLE_TYPE = SLOT>
bool q_poll(squeue<SLOT_TYPE>* q, VARIABLE_TYPE* s) {
  bool ret = q_poll_top_half<SLOT_TYPE, VARIABLE_TYPE>(q, s);
  if (ret) {
    read_barrier();
    q_poll_bottom_half<SLOT_TYPE>(q);
  }
  return ret;
}

// extern "C"
#endif

#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_DATA_H_
