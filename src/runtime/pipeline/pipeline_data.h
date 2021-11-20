#ifndef _PIPELINE_DATA_
#define _PIPELINE_DATA_
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>
#include <thread>
class ThreadPool1{
 public:
  ThreadPool1(){};
  int a = 0;
};
static ThreadPool1* ThreadLocal() { return dmlc::ThreadLocalStore<ThreadPool1>::Get(); }

void pipeline_run(int num){
  std::cout<<num <<std::endl;
  tvm::runtime::threading::Configure(tvm::runtime::threading::ThreadGroup::kSpecify,
      2, {0, 1, 2, 3}, 4);
  //auto t = ThreadLocal();
  //std::cout << "thread t is " << t << std::endl;
  tvm::runtime::threading::ResetThreadPool();
}
void test() {
  //return;
  //auto t = ThreadLocal();
  //auto t1 = ThreadLocal();
  tvm::runtime::threading::ResetThreadPool();
  auto td = std::thread(pipeline_run, 1);
  //std::cout << "loca t is " << t <<std::endl;
  td.join();
}
#endif
