#include <AEEStdErr.h>
#include <errno.h>
#include <inttypes.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "host/session.h"
#include "qwen3_probe.h"
#include "llama_mapping_probe.h"
#pragma weak rpcmem_alloc2
static uint64_t ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
int main(int argc,char **argv) {
 uint32_t count=argc>1?(uint32_t)strtoul(argv[1],NULL,10):7;
 uint32_t mib=argc>2?(uint32_t)strtoul(argv[2],NULL,10):768;
 uint32_t window=argc>3?(uint32_t)strtoul(argv[3],NULL,10):1;
 uint32_t cycles=argc>4?(uint32_t)strtoul(argv[4],NULL,10):20;
 uint32_t pin_kib=argc>5?(uint32_t)strtoul(argv[5],NULL,10):0;
 if(!count || count>LMP_MAX_BUFFERS || !mib || mib>1024 || !window || window>2 ||
    !cycles || (uint64_t)count*cycles>LMP_MAX_EVENTS || pin_kib>1048576 || !rpcmem_alloc2)return 2;
 setvbuf(stdout,NULL,_IOLBF,0);
 struct qbh_session session={(remote_handle64)-1,0};
 struct lmp_header *h=NULL;
 void *buf[LMP_MAX_BUFFERS]={0},*pin=NULL;
 int fds[LMP_MAX_BUFFERS],registered[LMP_MAX_BUFFERS]={0};
 int control_fd=-1,pin_fd=-1,control_mapped=0,pin_mapped=0,cleanup_errors=0,ret=1,run=-999;
 uint32_t size=mib*1048576U,pinbytes=pin_kib*1024U,host_bad=0;
 uint64_t alloc_ns=0,registration_ns=0,run_ns=0,t=ns();
 h=rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,sizeof(*h));
 if(!h)goto cleanup;
 memset(h,0,sizeof(*h));
 h->magic=LMP_MAGIC;h->version=1;h->count=count;h->window=window;h->cycles=cycles;h->pinned_bytes=pinbytes;
 for(uint32_t i=0;i<count;i++) {
  buf[i]=rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,size);
  if(!buf[i]){printf("{\"allocation_failed_buffer\":%u,\"errno\":%d}\n",i,errno);goto cleanup;}
  memset(buf[i],0,size);
  for(uint32_t s=0;s<LMP_SAMPLES;s++){
   uint32_t *v=(uint32_t *)((uint8_t *)buf[i]+lmp_offset(size,s));
   for(uint32_t w=0;w<512;w++)v[w]=lmp_value(i,s,w);
  }
  fds[i]=rpcmem_to_fd(buf[i]);h->fds[i]=fds[i];h->sizes[i]=size;
  printf("{\"allocated_buffer\":%u,\"bytes\":%u,\"fd\":%d}\n",i,size,fds[i]);
 }
 if(pinbytes){
  pin=rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,pinbytes);
  if(!pin)goto cleanup;
  memset(pin,0x5a,pinbytes);pin_fd=rpcmem_to_fd(pin);
 }
 alloc_ns=ns()-t;
 if(qbh_session_open(&session))goto cleanup;
 t=ns();
 control_fd=rpcmem_to_fd(h);
 if(fastrpc_mmap(CDSP_DOMAIN_ID,control_fd,h,0,sizeof(*h),FASTRPC_MAP_FD))goto cleanup;
 control_mapped=1;
 if(pinbytes){if(fastrpc_mmap(CDSP_DOMAIN_ID,pin_fd,pin,0,pinbytes,FASTRPC_MAP_FD))goto cleanup;pin_mapped=1;}
 for(uint32_t i=0;i<count;i++){
  int e=fastrpc_mmap(CDSP_DOMAIN_ID,fds[i],buf[i],0,size,FASTRPC_MAP_FD_DELAYED);
  printf("{\"register_buffer\":%u,\"result\":%d}\n",i,e);
  if(e)goto cleanup;
  registered[i]=1;
 }
 registration_ns=ns()-t;
 if(qbh_session_prepare(&session))goto cleanup;
 t=ns();run=qwen3_probe_run_llama_mapping(session.handle,control_fd,sizeof(*h));run_ns=ns()-t;
 for(uint32_t i=0;i<count;i++)for(uint32_t s=0;s<LMP_SAMPLES;s++){
  uint32_t *v=(uint32_t *)((uint8_t *)buf[i]+lmp_offset(size,s));
  for(uint32_t w=0;w<512;w++){
   if(v[w]!=lmp_value(i,s,w))host_bad++;
   if(v[w+512]!=(lmp_value(i,s,w)^cycles))host_bad++;
  }
 }
 if(pinbytes)for(uint32_t i=0;i<pinbytes;i+=4096)if(((uint8_t *)pin)[i]!=0x5a)host_bad++;
 for(uint32_t i=0;i<h->event_count && i<LMP_MAX_EVENTS;i++){
  struct lmp_event *e=&h->events[i];
  printf("{\"record\":\"mapping\",\"cycle\":%u,\"buffer\":%u,\"va\":%u,\"bytes\":%u,\"map_result\":%d,\"unmap_result\":%d,\"checks\":%u,\"mismatches\":%u,\"map_ticks\":%"PRIu64",\"dma_ticks\":%"PRIu64",\"unmap_ticks\":%"PRIu64"}\n",
   e->cycle,e->buffer,e->va,e->bytes,e->map_result,e->unmap_result,e->checks,e->mismatches,e->map_ticks,e->dma_ticks,e->unmap_ticks);
 }
 ret=run || h->status || h->cleanup_errors || host_bad || h->completed!=count*cycles;
 cleanup:
 for(uint32_t i=0;i<count;i++)if(registered[i])cleanup_errors+=fastrpc_munmap(CDSP_DOMAIN_ID,fds[i],buf[i],size)!=0;
 if(pin_mapped)cleanup_errors+=fastrpc_munmap(CDSP_DOMAIN_ID,pin_fd,pin,pinbytes)!=0;
 if(control_mapped)cleanup_errors+=fastrpc_munmap(CDSP_DOMAIN_ID,control_fd,h,sizeof(*h))!=0;
 if(session.handle!=(remote_handle64)-1)cleanup_errors+=qbh_session_close(&session)!=0;
 printf("{\"record\":\"summary\",\"count\":%u,\"buffer_bytes\":%u,\"resident_bytes\":%"PRIu64",\"window\":%u,\"cycles\":%u,\"pinned_bytes\":%u,\"run_result\":%d,\"dsp_status\":%d,\"completed\":%u,\"dsp_mismatches\":%u,\"host_mismatches\":%u,\"cleanup_errors\":%d,\"dsp_cleanup_errors\":%d,\"vtcm_bytes\":%u,\"allocation_ns\":%"PRIu64",\"registration_ns\":%"PRIu64",\"host_run_ns\":%"PRIu64",\"dsp_ticks\":%"PRIu64",\"pass\":%s}\n",
 count,size,(uint64_t)count*size+pinbytes,window,cycles,pinbytes,run,h?h->status:-999,h?h->completed:0,h?h->mismatches:0,host_bad,cleanup_errors,h?h->cleanup_errors:0,h?h->vtcm_bytes:0,alloc_ns,registration_ns,run_ns,h?h->total_ticks:0,(!ret&&!cleanup_errors)?"true":"false");
 for(uint32_t i=0;i<count;i++)if(buf[i])rpcmem_free(buf[i]);
 if(pin)rpcmem_free(pin);
 if(h)rpcmem_free(h);
 return ret||cleanup_errors?1:0;
}
