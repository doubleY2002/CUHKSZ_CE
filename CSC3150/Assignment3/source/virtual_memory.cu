#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define invalid 0x80000000

__device__ unsigned int head_tail=invalid;
__device__ int count;

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;

	  vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] = 0x80000000;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  u32 phy_addr=vm_find_phy_addr(vm,addr);

  uchar value=vm->buffer[phy_addr];

  return value; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 phy_addr=vm_find_phy_addr(vm,addr);
  vm->buffer[phy_addr]=value;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for(int i=0;i<input_size;i++){
    results[i]=vm_read(vm,i+offset);
  }
}

__device__ u32 vm_find_phy_addr(VirtualMemory *vm, u32 addr){
  u32 phy_addr;
  u32 storage_addr;

  int page=-1;
  for(int i=0;i<vm->PAGE_ENTRIES;i++){//Have been accessed?
    if(!(vm->invert_page_table[i]==invalid)){
      if(vm->invert_page_table[i+vm->PAGE_ENTRIES]==(addr/vm->PAGESIZE)){// page corresponding to the address
        page=i;
        break;
      }
    }
  }

  if(page!=-1){//Yes
    phy_addr=addr%vm->PAGESIZE+page*vm->PAGESIZE;
  }
  else{//No, shared memory space is available(empty)?

    (*vm->pagefault_num_ptr)++;

    for(int i=0;i<vm->PAGE_ENTRIES;i++){
      if(vm->invert_page_table[i]==invalid){//min index empty
        page=i;
        break;
      }
    }

    if(page!=-1){//Yes
      vm->invert_page_table[page]&=0x7FFFFFFF;//be accessed now
    }
    else{//No, replace the LRU set
      page=(int)((head_tail>>0)&0x3FF);
	    storage_addr=(vm->invert_page_table[page+vm->PAGE_ENTRIES])*vm->PAGESIZE;
      for(int i=0;i<vm->PAGESIZE;i++){
        vm->storage[storage_addr+i]=vm->buffer[page*vm->PAGESIZE+i];
      }
    }

    storage_addr=addr/vm->PAGESIZE*vm->PAGESIZE;
    vm->invert_page_table[page+vm->PAGE_ENTRIES]=addr/vm->PAGESIZE;
    for(int i=0;i<vm->PAGESIZE;i++){
      vm->buffer[page*vm->PAGESIZE+i]=vm->storage[storage_addr+i];
    }
    phy_addr=addr%vm->PAGESIZE+page*vm->PAGESIZE;
  }

  vm_LRU(vm,page);

  return phy_addr;
}

__device__ void vm_LRU(VirtualMemory *vm, int page){
	int LRU_head=(head_tail>>16)&0x3FF;
	int LRU_tail=(head_tail>> 0)&0x3FF;
	count++;
	// if(count%1024==0&&count>(1<<15))printf("%d %d      %d %d %d\n",count, (*vm->pagefault_num_ptr),page,LRU_head,LRU_tail);
	if(vm->invert_page_table[page+vm->PAGE_ENTRIES*2]==invalid){
		vm->invert_page_table[page+vm->PAGE_ENTRIES*2]=(0xFFFF)<<16;
		// printf("%d %d      %d %d %d\n",count, (*vm->pagefault_num_ptr),page,LRU_head,LRU_tail);

		if(head_tail==invalid){
			head_tail=0;
			head_tail|=page<<0;
			vm->invert_page_table[page+vm->PAGE_ENTRIES*2]|=(0xFFFF)<<0;
		}
		else{
			vm->invert_page_table[page+vm->PAGE_ENTRIES*2]&=0xFFFF0000;
			vm->invert_page_table[page+vm->PAGE_ENTRIES*2]|=LRU_head<<0;
			vm->invert_page_table[LRU_head+vm->PAGE_ENTRIES*2]&=0x0000FFFF;
			vm->invert_page_table[LRU_head+vm->PAGE_ENTRIES*2]|=page<<16;
			// printf("%d %d %d\n",page,LRU_head,(vm->invert_page_table[LRU_head+vm->PAGE_ENTRIES*2]>>16)&0x3ff);
		}
		head_tail&=0x0000FFFF;//change head, unchange tail
		head_tail|=page<<16;
		LRU_head=page;
	}

	if((page&0x3ff)==(int)LRU_head)return;
	else if((page&0x3ff)==(int)LRU_tail){
		int prev=(vm->invert_page_table[page+vm->PAGE_ENTRIES*2]>>16)&0x3FF;
		vm->invert_page_table[prev+vm->PAGE_ENTRIES*2]|=(0xFFFF)<<0;
		head_tail&=0xFFFF0000;
		head_tail|=prev<<0;
		// printf("orz%d",prev);
	}
	else{
		int prev=(vm->invert_page_table[page+vm->PAGE_ENTRIES*2]>>16)&0x3FF;
		int next=(vm->invert_page_table[page+vm->PAGE_ENTRIES*2]>>0)&0x3FF;

		vm->invert_page_table[prev+vm->PAGE_ENTRIES*2]&=0xFFFF0000;
		vm->invert_page_table[prev+vm->PAGE_ENTRIES*2]|=next<<0;
		vm->invert_page_table[next+vm->PAGE_ENTRIES*2]&=0x0000FFFF;
		vm->invert_page_table[next+vm->PAGE_ENTRIES*2]|=prev<<16;
		// printf("qwq");
	}

	vm->invert_page_table[LRU_head+vm->PAGE_ENTRIES*2]&=0x0000FFFF;
	vm->invert_page_table[LRU_head+vm->PAGE_ENTRIES*2]|=page<<16;
	
	vm->invert_page_table[page+vm->PAGE_ENTRIES*2]=0xFFFF0000;
	vm->invert_page_table[page+vm->PAGE_ENTRIES*2]|=LRU_head<<0;

	head_tail&=0x0000FFFF;
	head_tail|=page<<16;
}