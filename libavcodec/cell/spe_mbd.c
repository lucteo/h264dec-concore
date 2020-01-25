/*
 * Copyright (c) 2009 TUDelft
 *
 * Cell Parallel SPU - 2DWave Macroblock Decoding.
 */

/**
 * @file libavcodec/cell/spu/h264_main_spu.c
 * Cell Parallel SPU - 2DWave Macroblock Decoding
 * @author C C Chi <c.c.chi@student.tudelft.nl>
 *
 * SIMD kernels
 * H.264/AVC motion compensation
 * @author Mauricio Alvarez <alvarez@ac.upc.edu>
 * @author Albert Paradis <apar7632@hotmail.com>
 */


/* Enable this lines to enable simulator statistic or generate traces */

//#define ENABLE_SIMULATOR
//#define ENABLE_PARAVER_TRACING_CELL

#ifdef ENABLE_SIMULATOR
	#include "/opt/ibm/systemsim-cell/include/callthru/spu/profile.h"
#endif

#ifdef ENABLE_TRACES
	#include "spu_trace.h"
#endif
#include <unistd.h>
#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <libsync.h>
#include <sys/time.h>
#include <assert.h>

//#include "dsputil_cell.h"
#include "types_spu.h"
#include "h264_intra_spu.h"
#include "h264_decode_mb_spu.h"
#include "h264_mc_spu.h"
#include "h264_tables.h"
#include "h264_dma.h"


/** functions for supporting tracing with paraver for the SPU
 *
 */
inline void trace_init_SPU(){
#ifdef ENABLE_PARAVER_TRACING_CELL
	SPUtrace_init ();
#endif
}

inline void trace_fini_SPU(){
#ifdef ENABLE_PARAVER_TRACING_CELL
	SPUtrace_fini ();
#endif
}

inline void trace_event_SPU(int event, int id){
#ifdef ENABLE_PARAVER_TRACING_CELL
	SPUtrace_event (event, id);
#else
	(void) event;
	(void) id;
#endif
}

// for simulator statistic
inline void clear_statistic(){
#ifdef ENABLE_SIMULATOR
	prof_clear();
#endif
}

inline void start_statistic(){
#ifdef ENABLE_SIMULATOR
	prof_start();
#endif
}

inline void stop_statistic(){
#ifdef ENABLE_SIMULATOR
	prof_stop();
#endif
}

H264Context_spu h_context;  // struct that contain all the params to decode a macroblock

DECLARE_ALIGNED_16(spe_pos, dma_temp); //dma temp for sending
//mb position of neighbouring spes
DECLARE_ALIGNED_16(volatile spe_pos, src_spe); //written by SPE_ID -1
//DECLARE_ALIGNED_16(spe_pos, tgt_spe); //written by SPE_ID +1

/**
*	Initializes the buffering of the mb data and associated mc data. The init_mb_buffer needs to
*	be called before any get_next_mb and only once at the beginning of the slice.
*
*	Note: init_mc_buffer and get_next_mb expect the width of the picture to be more than 2 mb's
*/
#define TAG_OFFSET_MB MBD_buf1
#define TAG_OFFSET_MC MBD_mc_buf1
static void init_mb_buffer(H264Context_spu* h){
	H264slice *s = h->s;
	H264Mb *next_mb;
	int mb_height = s->mb_height;
	int mb_width = s->mb_width;

	h->mc_idx =0;

	h->mb_dec = 0;
	h->mb_mc = 0;
	h->mb_dma = 0;

	h->curr_line %= mb_height;
	h->next_mb_idx = h->curr_line * mb_width;
	h->mb_id = h->curr_line * mb_width;
	h->n_mc= h->curr_line * mb_width;

	next_mb = s->blocks + h->mb_id;
	spu_dma_get(&h->mb_buf[h->mb_dma], (unsigned) next_mb, sizeof(H264Mb), h->mb_dma + TAG_OFFSET_MB);
	h->mb_dma++;
	h->mb_id++;

	next_mb = s->blocks + h->mb_id;
	spu_dma_get(&h->mb_buf[h->mb_dma], (unsigned) next_mb, sizeof(H264Mb), h->mb_dma + TAG_OFFSET_MB);
	h->mb_dma++;
	h->mb_id++;
	wait_dma_id(0 + TAG_OFFSET_MB);

	H264Mb *mb = &h->mb_buf[0];
	H264mc *mc = &h->mc_buf[0];
	if(!IS_INTRA(mb->mb_type)){
		calc_mc_params(mb, mc);
		fill_ref_buf(h, mb, mc);
	}
	h->n_mc++;
	h->mb_mc++;
}

static void *get_next_mb(H264Context_spu *h){
	H264slice *s = h->s;
	H264spe *spe = &h->spe;
	H264Mb *mb_buf = h->mb_buf;
	H264mc *mc_buf = h->mc_buf;
	H264Mb *next_mb;
	H264Mb *next_dma_mb;

	if (h->curr_line >= s->mb_height)
		return NULL;

	if (h->mb_id < h->mb_total){
		next_dma_mb = s->blocks + h->mb_id;
		spu_dma_get(&mb_buf[h->mb_dma], (unsigned) next_dma_mb, sizeof(H264Mb), h->mb_dma + TAG_OFFSET_MB);
		h->mb_dma = (h->mb_dma+1)%3;
		h->mb_id++;
		if (h->mb_id%s->mb_width ==0){
			h->mb_id+=(spe->spe_total-1)*s->mb_width;
		}
	}

	h->mc = &mc_buf[h->mc_idx];
	wait_dma_id(h->mc_idx + TAG_OFFSET_MC);
	h->mc_idx = (h->mc_idx+1)%2;
	if (h->n_mc < h->mb_total){
		wait_dma_id(h->mb_mc + TAG_OFFSET_MB);
		H264Mb *mb = &mb_buf[h->mb_mc];
		H264mc *mc = &mc_buf[h->mc_idx];
		if(!IS_INTRA(mb->mb_type)){
			calc_mc_params(mb, mc);
			fill_ref_buf(h, mb, mc);
		}
		h->n_mc++;
		if (h->n_mc%s->mb_width ==0){
			h->n_mc+=(spe->spe_total-1)*s->mb_width;
		}
	}
	h->next_mb_idx++;
	if (h->next_mb_idx % s->mb_width ==0){
		h->next_mb_idx+=(spe->spe_total-1)*s->mb_width;
		h->curr_line+=spe->spe_total;
	}

	h->mb_mc = (h->mb_mc+1)%3;
	next_mb = &mb_buf[h->mb_dec];
	h->mb_dec = (h->mb_dec+1)%3;
	return next_mb;
}

static void *get_next_mb_blocking(H264Context_spu *h){
	H264slice *s = h->s;
	H264spe *spe = &h->spe;
	H264Mb *mb_buf = h->mb_buf;
	H264mc *mc_buf = h->mc_buf;
	H264Mb *next_mb;
	H264Mb *next_dma_mb;

	if (h->mb_id >= h->mb_total)
		return NULL;

	//printf("%d\n", h->mb_id);
	next_dma_mb = s->blocks + h->mb_id;
	spu_dma_get(&mb_buf[0], (unsigned) next_dma_mb, sizeof(H264Mb), MBD_buf1);
	//h->mb_dma = (h->mb_dma+1)%3;
	h->mb_id++;
	if (h->mb_id%s->mb_width ==0){
		h->mb_id+=(spe->spe_total-1)*s->mb_width;
	}
	wait_dma_id(MBD_buf1);

	h->mc = &mc_buf[0];
	//h->mc_idx = (h->mc_idx+1)%2;
	//if (h->n_mc < h->mb_total){
	H264Mb *mb = &mb_buf[0];
	H264mc *mc = &mc_buf[0];
	if(!IS_INTRA(mb->mb_type)){
		calc_mc_params(mb, mc);
		fill_ref_buf(h, mb, mc);
	}
	//h->n_mc++;
	/*if (h->n_mc%s->mb_width ==0){
		h->n_mc+=(spe->spe_total-1)*s->mb_width;
	}*/
//	wait_dma_id(MBD_mc_buf1);

// 	h->next_mb_idx++;
// 	if (h->next_mb_idx % s->mb_width ==0){
// 		h->next_mb_idx+=(spe->spe_total-1)*s->mb_width;
// 		h->curr_line+=spe->spe_total;
// 	}

// 	h->mb_mc = (h->mb_mc+1)%3;
	next_mb = &mb_buf[0];
// 	h->mb_dec = (h->mb_dec+1)%3;
	return next_mb;
}


#undef TAG_OFFSET_MB
#undef TAG_OFFSET_MC
static inline int dep_resolved(H264Context_spu *h){
	H264slice *s = h->s;
	int spe_id = h->spe.spe_id;
	volatile int mb_proc_dep = src_spe.count;
	if (spe_id==0)
		return (h->mb_proc < mb_proc_dep-1 +s->mb_width)? 1:0;
	else
		return (h->mb_proc < mb_proc_dep-1)? 1:0;
}

void update_tgt_spe_dep(H264Context_spu *h, int end){
	H264Mb *mb = h->mb;
	H264slice *s = h->s;
	H264spe *spe = &h->spe;
	int mb_x = mb->mb_x;

	if (end || (mb_x%2==0 && mb_x!=0) || mb_x==s->mb_width-1){
		spe_pos* dma_spe = &dma_temp;
		spe_pos* tgt_spe = (spe_pos*) ((unsigned) spe->tgt_spe + (unsigned) &src_spe); //located in target spe local store
		dma_spe->count = end? h->mb_proc+1: h->mb_proc;
		spu_dma_barrier_put(dma_spe, (unsigned) tgt_spe, sizeof(dma_temp), MBD_put);
	}
	h->mb_proc++;
}


int main(unsigned long long id, unsigned long long argp)
{
	(void) id;
	H264Context_spu* h = &h_context;
	H264spe *spe_params = (H264spe *) (unsigned) argp;

	spu_dma_get(&h->spe, (unsigned) spe_params, sizeof(H264spe), MBD_slice); //ID_slice is used out of convienience
	wait_dma_id(MBD_slice);

    //clear_statistic();
    dsputil_h264_init_cell(&h->dsp);
    ff_cropTbl_init();
    init_pred_ptrs(&h->hpc);

	//send slice_buf to ppe
	spu_write_out_mbox((unsigned) h->slice_buf);
	h->sl_idx=0;
	// initialize tracing with paraver
    //trace_init_SPU();
	h->frames =0;
	src_spe.count =0;
	h->mb_proc = 0;

	h->mb_id=0;
	h->mc_idx=0;
	h->mb_dec=0;
	h->mb_mc=0;
	h->mb_dma=0;
	h->next_mb_idx=0;

	h->blocking=0;


	H264spe* p = &h->spe;
	h->curr_line =p->spe_id;
	h->mb_total = p->mb_height*p->mb_width;
	int stride_y = 32;
	int stride_c = 16;
	//init block_offset array
	init_block_offset(stride_y, stride_c);
	for(;;){
		spu_read_in_mbox();

		h->s = &h->slice_buf[h->sl_idx];
		h->sl_idx++; h->sl_idx%=2;

		if (h->s->state< 0){
			break;
		}

		{
			if(!h->blocking){
				init_mb_buffer(h);
				while((h->mb=(H264Mb *)get_next_mb(h))){
					while(!dep_resolved(h));
					//printf("frame %d mbx %d\t mby %d id %d\n", h->frames, h->mb->mb_x, h->mb->mb_y, p-	>spe_id);
					hl_decode_mb_internal(h, stride_y, stride_c);
				}
				update_tgt_spe_dep(h, 1);
			}else{
				h->mb_id=0;
				while((h->mb=(H264Mb *)get_next_mb_blocking(h))){
					while(!dep_resolved(h));
					//printf("frame %d mbx %d\t mby %d id %d\n", h->frames, h->mb->mb_x, h->mb->mb_y, p-	>spe_id);
					hl_decode_mb_internal(h, stride_y, stride_c);
				}
				update_tgt_spe_dep(h, 1);
			}

		}

		h->frames++;

		if (p->spe_id == ((h->frames*p->mb_height -1)%p->spe_total)){
			//printf("spe %d, %d\n", atomic_read(p->rl_cnt), h->frames);
			//MBSlice is copied beforehand.
			//only inc cnt.
			atomic_inc(p->rl_cnt);
		}
		{
			atomic_dec(p->cnt);
		}
	}

	return 0;
}

