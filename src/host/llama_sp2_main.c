#include <AEEStdErr.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "host/session.h"
#include "qwen3_probe.h"
#include "llama_sp2_probe.h"
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;}
int main(int argc,char **argv){
 if(argc!=3){fprintf(stderr,"usage: llama_sp2_cli input.bin output.bin\n");return 2;}
 FILE *f=fopen(argv[1],"rb");if(!f)return 2;fseek(f,0,SEEK_END);long size=ftell(f);rewind(f);
 if(size<128 || size>16777216){fclose(f);return 2;}
 uint8_t *shared=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,(int)size);if(!shared){fclose(f);return 2;}
 if(fread(shared,1,size,f)!=(size_t)size){fclose(f);rpcmem_free(shared);return 2;}fclose(f);
 struct qbh_session session={(remote_handle64)-1,0};int fd=rpcmem_to_fd(shared),mapped=0,ret=1;uint64_t elapsed=0;
 if(fd<0 || qbh_session_open(&session) || qbh_session_prepare(&session))goto done;
 if(fastrpc_mmap(CDSP_DOMAIN_ID,fd,shared,0,size,FASTRPC_MAP_FD))goto done;mapped=1;
 ret=qwen3_probe_run_llama_sp2(session.handle,fd,(uint32_t)size);if(ret)goto done;
 elapsed=now();ret=qwen3_probe_run_llama_sp2(session.handle,fd,(uint32_t)size);elapsed=now()-elapsed;
 f=fopen(argv[2],"wb");if(!f){ret=1;goto done;}if(fwrite(shared,1,size,f)!=(size_t)size)ret=1;fclose(f);
 {struct lsp2_header *h=(void*)shared;printf("{\"rpc_status\":%d,\"dsp_status\":%d,\"host_ns\":%llu,\"mode\":%u,\"rows\":%u,\"k\":%u,\"n\":%u,\"vtcm_bytes\":%u,\"peak_bytes\":%u,\"streams\":%u,\"conversions\":%u,\"total_ticks\":%llu,\"load_ticks\":%llu,\"pack_ticks\":%llu,\"dma_ticks\":%llu,\"mac_ticks\":%llu,\"convert_ticks\":%llu,\"merge_ticks\":%llu,\"publish_ticks\":%llu}\n",ret,h->status,(unsigned long long)elapsed,h->mode,h->rows,h->k,h->n,h->vtcm_bytes,h->peak_bytes,h->streams,h->conversions,(unsigned long long)h->total_ticks,(unsigned long long)h->load_ticks,(unsigned long long)h->pack_ticks,(unsigned long long)h->dma_ticks,(unsigned long long)h->mac_ticks,(unsigned long long)h->convert_ticks,(unsigned long long)h->merge_ticks,(unsigned long long)h->publish_ticks);}
 done:
 if(mapped)fastrpc_munmap(CDSP_DOMAIN_ID,fd,shared,size);
 qbh_session_close(&session);rpcmem_free(shared);return ret?1:0;
}
