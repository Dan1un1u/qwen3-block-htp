#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <rpcmem.h>
#include "host/session.h"
#include "qwen3_probe.h"
#include "bandwidth_protocol.h"
int main(int argc,char **argv) {
 if(argc!=8) {fprintf(stderr,"mode bytes workers repeats stream depth bypass\n");return 2;}
 struct qbh_session s={(remote_handle64)-1,0};
 uint32_t size=BW_PAYLOAD+BW_DRAM;
 uint8_t *mem=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,(int)size);
 if(!mem)return 3;
 struct bw_header *h=(void*)mem;memset(h,0,sizeof(*h));h->magic=BW_MAGIC;
 h->mode=strtoul(argv[1],0,0);h->bytes=strtoul(argv[2],0,0);h->workers=strtoul(argv[3],0,0);
 h->repeats=strtoul(argv[4],0,0);h->stream=strtoul(argv[5],0,0);h->depth=strtoul(argv[6],0,0);h->bypass=strtoul(argv[7],0,0);h->rounds=BW_ROUNDS;
 /* Address-dependent, deterministic DDR pattern; init is outside timing. */
 if(h->mode==6)for(uint32_t i=0;i<BW_DRAM;i++)mem[BW_PAYLOAD+i]=(uint8_t)((i*13U+(i>>12)*7U)^0x5aU);
 int fd=rpcmem_to_fd(mem),rc=qbh_session_open(&s),mapped=0;
 if(!rc){rc=fastrpc_mmap(CDSP_DOMAIN_ID,fd,mem,0,size,FASTRPC_MAP_FD);mapped=!rc;}
 if(!rc)rc=qbh_session_prepare(&s);
 if(!rc)rc=qwen3_probe_run_bandwidth(s.handle,fd,size);
 printf("{\"mode\":%u,\"bytes\":%u,\"workers\":%u,\"repeats\":%u,\"stream\":%u,\"depth\":%u,\"bypass\":%u,\"rc\":%d,\"status\":%u,\"errors\":%u,\"vtcm\":%u,\"payload\":%"PRIu64",\"ticks\":[",h->mode,h->bytes,h->workers,h->repeats,h->stream,h->depth,h->bypass,rc,h->status,h->errors,h->vtcm_bytes,h->payload_bytes);
 for(unsigned i=0;i<h->rounds;i++)printf("%s%"PRIu64,i?",":"",h->ticks[i]);
 printf("],\"hvx_units\":%u,\"cycles\":[",h->checks[0]);for(unsigned i=0;i<h->rounds;i++)printf("%s%"PRIu64,i?",":"",h->cycles[i]);printf("]}\n");
 int fail=rc||h->status||h->errors;qbh_session_close(&s);if(mapped)fastrpc_munmap(CDSP_DOMAIN_ID,fd,mem,size);rpcmem_free(mem);return fail?1:0;
}
