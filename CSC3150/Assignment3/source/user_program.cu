﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//                              int input_size) {
//   for (int i = 0; i < input_size; i++)//input_size 128KB 4*(32*1024)
//     vm_write(vm, i, input[i]);//4096 pages, all new to LRU, page_fault+=4096

//   for (int i = input_size - 1; i >= input_size - 32769; i--)//input_size-1 to input_size - 32768 is in LRU
//     int value = vm_read(vm, i);//page_fault+=1  (input_size-32769)

//   vm_snapshot(vm, results, 0, input_size);//input_size 128KB 4*(32*1024)
//   //4096 pages, as LRU pages is update, all new
// }

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
  int input_size) {
// write the data.bin to the VM starting from address 32*1024
  for (int i = 0; i < input_size; i++)//4096 pages
    vm_write(vm, 32*1024+i, input[i]);
// write (32KB-32B) data  to the VM starting from 0
  for (int i = 0; i < 32*1023; i++)//1023 pages
    vm_write(vm, i, input[i+32*1024]);
// readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
  vm_snapshot(vm, results, 32*1024, input_size);//4096 pages (LRU is 0-1022, 5120)
}

// // expected page fault num: 9215