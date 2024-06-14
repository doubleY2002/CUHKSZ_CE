#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define valid_bit(i) (i) * fs->FCB_SIZE + 28 + fs->SUPERBLOCK_SIZE
#define size_bit(i)  (i) * fs->FCB_SIZE + 26 + fs->SUPERBLOCK_SIZE
#define addr_bit(i)  (i) * fs->FCB_SIZE + 24 + fs->SUPERBLOCK_SIZE
#define mtime_bit(i) (i) * fs->FCB_SIZE + 22 + fs->SUPERBLOCK_SIZE
#define ctime_bit(i) (i) * fs->FCB_SIZE + 20 + fs->SUPERBLOCK_SIZE
#define name_bit(i)  (i) * fs->FCB_SIZE + 0  + fs->SUPERBLOCK_SIZE

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 file_number = 0;
__device__ __managed__ u32 gsize = 0; // last block number

__device__ void FCB_init(FileSystem *fs)
{
  // 0-19 name
  // 20-21 create time
  // 22-23 modified time
  // 24-25 address
  // 26-27 size
  // 28 valid bit
  for (int i=0;i<fs->FCB_ENTRIES;i++)//set valid bit
  {
    fs->volume[valid_bit(i)] = 0xff;
  }
}
__device__ void SUPERBLOCK_init(FileSystem *fs)
{
  for (int i=0;i<fs->SUPERBLOCK_SIZE;i++)
  {
    fs->volume[i]=0;
  }
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  SUPERBLOCK_init(fs);
  FCB_init(fs);
}

__device__ void string_print(char *s)
{
  while(*s != '\0'){
    printf("%c",*s);
    s++;
  }
  // printf("\n");
}

__device__ bool check_name(char *s1, char *s2)
{
  while(*s1 != '\0' && *s2 != '\0' && *s1 == *s2)
  {
    s1++;
    s2++;
  }

  if((*s1 == '\0')&&(*s2 == '\0'))
  {
    return true;
  }
  return false;
}


__device__ bool update_file_name(FileSystem *fs, char *s, int file)
{
  int count=0;
  while(*s != '\0')
  {
    fs->volume[name_bit(file)+count] = *s;
    s++;
    count++;
    if(count == fs->MAX_FILENAME_SIZE)
    {
      printf("ERROR: file name too large.\n");
      return true;
    }
  }
  fs->volume[name_bit(file)+count] = '\0';
  return false;
}
__device__ void modified_FCB(FileSystem *fs, int file, int size)
{
  //modified time
  fs->volume[mtime_bit(file)+0] = (gtime>>8) & 0xff;
  fs->volume[mtime_bit(file)+1] = (gtime>>0) & 0xff;

  //modified size
  fs->volume[size_bit(file)+0] = (size>>8) & 0xff;
  fs->volume[size_bit(file)+1] = (size>>0) & 0xff;

  //valid bit set to 0
  fs->volume[valid_bit(file)] = 0;
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	int empty = -1;
  for(int i = 0; i < fs->FCB_ENTRIES; i++)// find file
  {
    if(fs->volume[valid_bit(i)] != 0xff)// valid? Yes
    {
      if(check_name((char *)&fs->volume[name_bit(i)], s))// The file? Yes
      {
        return i;
      }
    }
    else // valid? No
    {
      if(empty == -1)// First empty address
      {
        empty = i;
      }
    }
  }

//  if(op == G_READ)printf("ERROR: No such file.\n");

  if(empty == -1)// cant build another file
  {
    printf("ERROR: files number reach the maximun.\n");
    return 0xffffffff;
  }
  else//create new file to write
  {
    if(update_file_name(fs, s, empty))// cant write file name
    {
      return 0xffffffff;
    }
    modified_FCB(fs, empty, 0);

    fs->volume[ctime_bit(empty)+0] = (gtime>>8) & 0xff;
    fs->volume[ctime_bit(empty)+1] = (gtime>>0) & 0xff;

    gtime++;
    file_number++;
  }
  return empty;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	if( fp >= 1024 && fp < 0 && fs->volume[valid_bit(fp)] == 0xff)
  {
    printf("ERROR: error fp.\n");
    return;
  }

  int file_size = (fs->volume[size_bit(fp)+0]<<8) + fs->volume[size_bit(fp)+1];
  int file_addr = (fs->volume[addr_bit(fp)+0]<<8) + fs->volume[addr_bit(fp)+1];
  file_addr *= fs->STORAGE_BLOCK_SIZE;

  if(file_size < size)
  {
    // uchar *s = &fs->volume[file_addr + fs->FILE_BASE_ADDRESS];
    // printf("fp=%d size = %d require size = %d   ",(int)fp, file_size, size);
    // string_print((char *)s);

    printf("ERROR: require size is too large\n");
    return;
  }

  for(int i=0; i < size; i++)
  {
    output[i] = fs->volume[file_addr + i + fs->FILE_BASE_ADDRESS]; 
  }
}

__device__ void set_superblock_bit(FileSystem *fs, u32 addr, int bit)
{
  int i = addr >> 3;
  int j = addr & 0x07;
  u_int update = (1<<j);
  if(bit==1)
  {
    fs->volume[i] |= update;
  }
  else
  {
    fs->volume[i] &= ~(update);
  }
}

__device__ void remove_file(FileSystem *fs, u32 fp)
{
  int size = (fs->volume[size_bit(fp)+0]<<8) + fs->volume[size_bit(fp)+1];
  int addr = (fs->volume[addr_bit(fp)+0]<<8) + fs->volume[addr_bit(fp)+1];
  addr *= fs->STORAGE_BLOCK_SIZE;

  for (int i = 0; i < size; i++)//release fp addr
  {
    fs->volume[i + addr + fs->FILE_BASE_ADDRESS] = 0;
  }
  int block_number = (size+fs->STORAGE_BLOCK_SIZE-1) / fs->STORAGE_BLOCK_SIZE;
  addr /= fs->STORAGE_BLOCK_SIZE;
  for (int i = 0; i < block_number; i++)// set superblock of fp to 0
  {
    set_superblock_bit(fs, addr+i, 0);
  }

  for (int i = addr + block_number; i < gsize; i++)// move addr after fp block_number front
  {
    for (int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++)// size
    {
      fs->volume[(i-block_number)*fs->STORAGE_BLOCK_SIZE+j+fs->FILE_BASE_ADDRESS] = fs->volume[i*fs->STORAGE_BLOCK_SIZE+j+fs->FILE_BASE_ADDRESS];
    }
    set_superblock_bit(fs, i-block_number, 1);
    set_superblock_bit(fs, i, 0);
  }

  for (int i = 0; i < fs->FCB_ENTRIES; i++)//search file
  {
    if(fs->volume[valid_bit(i)]!=0xff)// the file vaild
    {
      int i_addr = (fs->volume[addr_bit(i)+0]<<8) + fs->volume[addr_bit(i)+1];
      if(i_addr > addr)// reset addr pointer of other file (after fp)
      {
        i_addr -= block_number;
        fs->volume[addr_bit(i)+0] = (i_addr>>8) & 0xff;
        fs->volume[addr_bit(i)+1] = (i_addr>>0) & 0xff;
      }
    }
  }
  gsize -= block_number;// total number 
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	if( fp >= 1024 && fp < 0 && fs->volume[valid_bit(fp)] == 0xff)
  {
    printf("ERROR: error fp.\n");
    return 0xffffffff;
  }
  if(size > fs->MAX_FILE_SIZE)
  {
    printf("ERROR: file size too large.\n");
    return 0xffffffff;
  }
  
  int old_size = (fs->volume[size_bit(fp)+0]<<8) + fs->volume[size_bit(fp)+1];
  int addr;
  
  if (old_size == 0)// set in end
  {
    addr = gsize;
  }
  else if((old_size + fs->STORAGE_BLOCK_SIZE-1)%32 != (size+ fs->STORAGE_BLOCK_SIZE-1)%32)//storage space not change
  {
    addr = (fs->volume[addr_bit(fp)+0]<<8) + fs->volume[addr_bit(fp)+1];
  }
  else// cant fit in well
  {
    remove_file(fs, fp);
    addr = gsize;
  }
  /*reset modified time*/
  gtime++;
  fs->volume[mtime_bit(fp)+0] = (gtime>>8) & 0xff;
  fs->volume[mtime_bit(fp)+1] = (gtime>>0) & 0xff;
  
  /*reset address*/
  fs->volume[addr_bit(fp)+0] = (addr>>8) & 0xff;
  fs->volume[addr_bit(fp)+1] = (addr>>0) & 0xff;
  
  /*reset size*/
  fs->volume[size_bit(fp)+0] = (size>>8) & 0xff;
  fs->volume[size_bit(fp)+1] = (size>>0) & 0xff;
  int block_number = (size + fs->STORAGE_BLOCK_SIZE-1)/fs->STORAGE_BLOCK_SIZE;
  for (int i = 0; i < block_number; i++)//superblock valid
  {
    set_superblock_bit(fs, addr+i, 1);
  }
  if(gsize == addr) gsize += block_number;

  addr *= fs->STORAGE_BLOCK_SIZE;
  for (int i = 0; i < size ; i++)//storage input
  {
    fs->volume[addr + i + fs->FILE_BASE_ADDRESS] = input[i];
  }

  // printf("fp=%d size=%d %d\n",(int)fp,size,(fs->volume[size_bit(fp)+0]<<8) + fs->volume[size_bit(fp)+1]);
  // printf("gsize=%d\n",gsize);
}

__device__ bool cmp(FileSystem *fs, int a, int b, int op)
{
  if (a == -1) return true;
  if (op == LS_D)
  {
    int a_time = (fs->volume[mtime_bit(a)+0]<<8) + (fs->volume[mtime_bit(a)+1]);
    int b_time = (fs->volume[mtime_bit(b)+0]<<8) + (fs->volume[mtime_bit(b)+1]);
    if(a_time != b_time) return (a_time < b_time);
  }
  else 
  {
    int a_size = (fs->volume[size_bit(a)+0]<<8) + (fs->volume[size_bit(a)+1]);
    int b_size = (fs->volume[size_bit(b)+0]<<8) + (fs->volume[size_bit(b)+1]);
    if (a_size != b_size) return (a_size < b_size);
  }

  int a_ctime = (fs->volume[ctime_bit(a)+0]<<8) + (fs->volume[ctime_bit(a)+1]);
  int b_ctime = (fs->volume[ctime_bit(b)+0]<<8) + (fs->volume[ctime_bit(b)+1]);
  return a_ctime > b_ctime;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == LS_D)
  {
    printf("===sort by modified time===\n");
  }
  if (op == LS_S)
  {
    printf("===sort by file size===\n");
  }
  for (int cnt = 0; cnt < file_number; cnt++)
  {
    int first=-1;
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      if (fs->volume[valid_bit(i)] == 0)
      {
        if (cmp(fs, first, i, op))
        {
          first=i;
        }
      }
    }
    if (first != -1)
    {
      string_print((char*)&fs->volume[name_bit(first)]);
      if (op == LS_S) printf(" %d",(fs->volume[size_bit(first)]<<8)+(fs->volume[size_bit(first)+1]));
      printf("\n");
      fs->volume[valid_bit(first)]=0x01;
    }
  }

  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    if(fs->volume[valid_bit(i)]==0x01)
    {
      fs->volume[valid_bit(i)]=0x00;
    }
  }
  return;
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  int fp = -1;
  for(int i = 0; i < fs->FCB_ENTRIES; i++)// find file
  {
    if(fs->volume[valid_bit(i)] != 0xff)// valid? Yes
    {
      if(check_name((char *)&fs->volume[name_bit(i)], s))// The file? Yes
      {
        fp = i;
        break;
      }
    }
  }
  if (fp == -1)
  {
    printf("ERROR: haven't find the file.");
    return;
  }
  remove_file(fs, (u32)fp);
  fs->volume[valid_bit(fp)]=0xff;
}
