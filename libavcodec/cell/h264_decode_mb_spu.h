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

#ifndef H264_DECODE_MB_SPU_H
#define H264_DECODE_MB_SPU_H

#define CELL_SPE
#include "libavcodec/avcodec.h"
#include "types_spu.h"
#include "h264_types_spu.h"
#include "h264_mc_spu.h"
#include "h264_dma.h"
#include "dsputil_spu.h"
#include "h264_intra_spu.h"

/**
 * H264Context
 */
typedef struct H264Context_spu{
	DECLARE_ALIGNED_16(H264spe, spe);		// contains simple type parameters that doesn't change
    DECLARE_ALIGNED_16(H264Mb, mb_buf[3]);			// contains simple type parameters that changes for macroblock
    DECLARE_ALIGNED_16(H264slice, slice_buf[2]);	// contains simple type parameters that changes for slice
	
	DSPContext_spu dsp;  // struct that contains pointers to mc interpolations functions
	H264PredContext_spu hpc;  // struct that contains pointers to intra prediction functions

	H264slice *s;
	int sl_idx;
	int frames;
	//mc arg buffer
	H264mc mc_buf[2];
	H264mc *mc;		//mc ptr to current decoded mb
	int mc_idx;
	int n_mc;		//next mb_id to mc
	int mb_proc;
	int mb_total;
	int curr_line;
	
	H264Mb* mb;		//mb ptr to current decoded mb
	int mb_id;		//next mb_id to dma
	int mb_dec; 	//mb_buf index - decoded mb
	int mb_mc;		//mb_buf index - prebuffer motion data
	int mb_dma;		//mb_buf index - target for dma mb data
	int next_mb_idx;
/*// for deblocking filter
    int edges[2];
    int start[2]; 
    int bS[2][4][4];				// dir, edge, bS;
    int qp[2][4];					// dir, edge;
    int chroma_qp[2][2][4];			// cb/cr, dir, edge;	
*/
	int blocking; 
}H264Context_spu;

void print_output(H264Context_spu* h, const char* msg);
void hl_decode_mb_internal(H264Context_spu *h, int stride_y, int stride_c);
void update_tgt_spe_dep(H264Context_spu *h, int end);

// IDCT functions
void (*idct_add)(uint8_t *dst, DCTELEM *block, int stride);
void (*idct_dc_add)(uint8_t *dst, DCTELEM *block, int stride);

void ff_idct_dc_add(uint8_t *dst, DCTELEM *block, int stride);
void ff_idct8_dc_add(uint8_t *dst, DCTELEM *block, int stride);

void ff_cropTbl_init();
void add_pixels8_c(uint8_t *pixels, DCTELEM *block, int line_size);
void add_pixels4_c(uint8_t *pixels, DCTELEM *block, int line_size);
void chroma_dc_dequant_idct_c(DCTELEM *block, int qmul);
void h264_luma_dc_dequant_idct_c(DCTELEM *block, int qmul);
// Filter functions
//void calculate_bS_qp(H264Context_spu *h);

// Motion compensation function
void fill_ref_buf(H264Context_spu *h, H264Mb *mb, H264mc *mc);
void calc_mc_params(H264Mb *mb, H264mc *mc);
void hl_motion(H264Context_spu *h, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c);


// Function to get traces
void trace_event_SPU(int event, int id);

#endif
