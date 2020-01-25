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

#include <stdio.h>
#include <string.h>
#include <spu_intrinsics.h>
//#include "dsputil_cell.h"
#include "types_spu.h"
#include "h264_tables.h"
#include "h264_dma.h"
#include "h264_mc_spu.h"
#include "h264_intra_spu.h"
#include "h264_decode_mb_spu.h"
#include "h264_deblock_spu.h"

//border buffers
DECLARE_ALIGNED_16(TopBorder, top_ls[240]);
LeftBorder left_ls;

//mb line buffer - statically allocated for up to 1920 width video
DECLARE_ALIGNED_16(uint8_t, dest_y_ls[2*16*20]);
DECLARE_ALIGNED_16(uint8_t, dest_cb_ls[2*8*10]);
DECLARE_ALIGNED_16(uint8_t, dest_cr_ls[2*8*10]);

//dma transfer buffer
DECLARE_ALIGNED_16(uint8_t, dma_y_ls [64*(32+20)]); //EDGE_WIDTH = 32
DECLARE_ALIGNED_16(uint8_t, dma_cb_ls[32*(16+10)]);
DECLARE_ALIGNED_16(uint8_t, dma_cr_ls[32*(16+10)]);

DECLARE_ALIGNED_16(uint8_t, extra_edge_y [32*(32+20)]); //EDGE_WIDTH = 32
DECLARE_ALIGNED_16(uint8_t, extra_edge_cr[16*(16+10)]);
DECLARE_ALIGNED_16(uint8_t, extra_edge_cb[16*(16+10)]);


// For intra mode
/// for now do the extra copy before dma, but it's better to skip this and do the dma right away
static void backup_mb_border(H264Context_spu *h, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize){
	H264Mb* mb= h->mb;
	
    int i;
	uint8_t* top_border_y = top_ls[mb->mb_x].unfiltered_y;
	uint8_t* top_border_cb = top_ls[mb->mb_x].unfiltered_cb;
	uint8_t* top_border_cr = top_ls[mb->mb_x].unfiltered_cr;
	
	uint8_t* left_border_y = left_ls.unfiltered_y;
	uint8_t* left_border_cb = left_ls.unfiltered_cb;
	uint8_t* left_border_cr = left_ls.unfiltered_cr;
		
    src_y  -=   linesize;
    src_cb -= uvlinesize;
    src_cr -= uvlinesize;

    // There are two lines saved, the line above the top macroblock of a pair,
    // and the line above the bottom macroblock
    left_border_y[0] = top_border_y[15];
    for(i=1; i<17; i++){
        left_border_y[i] = src_y[15+i*  linesize];
    }

   *(qword*)(top_border_y)= *(qword*)(src_y +  16*linesize);

    left_border_cb[0] = top_border_cb[7];
    left_border_cr[0] = top_border_cr[7];
    for(i=1; i<9; i++){
        left_border_cb[i] = src_cb[7+i*uvlinesize];
        left_border_cr[i] = src_cr[7+i*uvlinesize];
    }
    *(uint64_t*)(top_border_cb)= *(uint64_t*)(src_cb+8*uvlinesize);
    *(uint64_t*)(top_border_cr)= *(uint64_t*)(src_cr+8*uvlinesize);
}

static void xchg_mb_border(H264Context_spu *h, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize, int xchg){
	H264Mb* mb= h->mb;
	H264slice* s = h->s;
	
	int temp8, i;
	uint64_t temp64;
	int deblock_left;
	int deblock_top;
	
	uint8_t* top_border_y = top_ls[mb->mb_x].unfiltered_y;	
	uint8_t* top_border_cb = top_ls[mb->mb_x].unfiltered_cb;
	uint8_t* top_border_cr = top_ls[mb->mb_x].unfiltered_cr;
	uint8_t* top_border_y_next = top_ls[mb->mb_x +1].unfiltered_y;
	
	uint8_t* left_border_y = left_ls.unfiltered_y;
	uint8_t* left_border_cb = left_ls.unfiltered_cb;
	uint8_t* left_border_cr = left_ls.unfiltered_cr;
	
	deblock_left = (mb->mb_x > 0);
	deblock_top =  (mb->mb_y > 0);
	
	src_y  -= (  linesize + 1);
	src_cb -= (uvlinesize + 1);
	src_cr -= (uvlinesize + 1);
	
	#define XCHG(a,b,t,xchg)\
	t= a;\
	if(xchg)\
		a= b;\
	b= t;
	
	if(deblock_left){
		for(i = !deblock_top; i<16; i++){
			XCHG(left_border_y[i], src_y [i*  linesize], temp8, xchg);
		}
		XCHG(left_border_y[i], src_y [i*  linesize], temp8, 1);
		
		for(i = !deblock_top; i<8; i++){
			XCHG(left_border_cb[i], src_cb[i*uvlinesize], temp8, xchg);
			XCHG(left_border_cr[i], src_cr[i*uvlinesize], temp8, xchg);
		}
		XCHG(left_border_cb[i], src_cb[i*uvlinesize], temp8, 1);
		XCHG(left_border_cr[i], src_cr[i*uvlinesize], temp8, 1);
	}
	
	if(deblock_top){
		XCHG(*(uint64_t*)(top_border_y+0), *(uint64_t*)(src_y +1), temp64, xchg);
		XCHG(*(uint64_t*)(top_border_y+8), *(uint64_t*)(src_y +9), temp64, 1);
		if(mb->mb_x+1 < s->mb_width){
			XCHG(*(uint64_t*)(top_border_y_next), *(uint64_t*)(src_y +17), temp64, 1);
		}
		XCHG(*(uint64_t*)(top_border_cb), *(uint64_t*)(src_cb+1), temp64, 1);
		XCHG(*(uint64_t*)(top_border_cr), *(uint64_t*)(src_cr+1), temp64, 1);
	}
}

void copy_top_borders(int mb_x, uint8_t *dst_y, uint8_t *dst_cb, uint8_t *dst_cr, int stride_y, int stride_c){			
	qword *qsrc_y = (qword *) (top_ls[mb_x].top_borders_y);
	dst_y-= 4*stride_y;
	
	*((qword *) (dst_y + 0*stride_y)) = *qsrc_y++;
	*((qword *) (dst_y + 1*stride_y)) = *qsrc_y++;
	*((qword *) (dst_y + 2*stride_y)) = *qsrc_y++;
	*((qword *) (dst_y + 3*stride_y)) = *qsrc_y++;

	dst_cb-=2*stride_c;	
	uint64_t *dsrc_cb = (uint64_t *) (top_ls[mb_x].top_borders_cb);
	*((uint64_t *) (dst_cb + 0*stride_c)) = *dsrc_cb++; 
	*((uint64_t *) (dst_cb + 1*stride_c)) = *dsrc_cb++;

	dst_cr-=2*stride_c;	
	uint64_t *dsrc_cr = (uint64_t *) (top_ls[mb_x].top_borders_cr);
	*((uint64_t *) (dst_cr + 0*stride_c)) = *dsrc_cr++;
	*((uint64_t *) (dst_cr + 1*stride_c)) = *dsrc_cr++;
}

static void send_top_borders(H264Context_spu *h, int mb_x, uint8_t* dest_y, uint8_t* dest_cb, uint8_t* dest_cr, int stride_y, int stride_c){
	H264spe *spe= &h->spe;
	//fill borders (unfiltered borders already filled in backup_mb_border)
	dest_y+= 12*stride_y;
	qword *qtop_y = (qword *) top_ls[mb_x].top_borders_y;	
	for(int i=0; i<4; i++){
		qword *qdest_y = (qword *) dest_y;
		*qtop_y++ = *qdest_y;		
		dest_y+=stride_y;
	}
	dest_cb+= 6*stride_c;
	dest_cr+= 6*stride_c;
	uint64_t *dtop_cb = (uint64_t *) top_ls[mb_x].top_borders_cb;
	uint64_t *dtop_cr = (uint64_t *) top_ls[mb_x].top_borders_cr;
	for(int i=0; i<2; i++){
		uint64_t *ddest_cb = (uint64_t *) dest_cb;
		uint64_t *ddest_cr = (uint64_t *) dest_cr;
		
		*dtop_cb++  = *ddest_cb;
		*dtop_cr++  = *ddest_cr;
		
		dest_cb+=stride_c;
		dest_cr+=stride_c;
	}
	uint8_t* top_border_tgt = spe->tgt_spe + (unsigned) &top_ls[mb_x];
	spu_dma_put(&top_ls[mb_x], (unsigned) top_border_tgt, sizeof(TopBorder), MBD_put);
}

static void extend_edges_left(uint8_t *dma_y, uint8_t *dma_cb, uint8_t *dma_cr , int lines, int lines_c){
	for (int i=0; i<lines; i++){
		memset(dma_y, dma_y[32], 32);
		dma_y+=64;
	}

	for (int i=0; i<lines_c; i++){
		memset(dma_cb, dma_cb[16], 16);
		memset(dma_cr, dma_cr[16], 16);
		dma_cb+=32; dma_cr+=32;
	}
}

static void extend_edges_right(uint8_t *dma_y, uint8_t *dma_cb, uint8_t *dma_cr , int lines, int lines_c, int slots){
		
	for (int i=0; i<lines; i++){
		memset(dma_y, dma_y[-1], slots*16);
		dma_y+=64;
	}
	
	for (int i=0; i<lines_c; i++){
		memset(dma_cb, dma_cb[-1], slots*8);
		memset(dma_cr, dma_cr[-1], slots*8);
		dma_cb+=32; dma_cr+=32;
	}
}

static void extend_edges_top(uint8_t *dma_y, uint8_t *dma_cb, uint8_t *dma_cr ){
	qword *qborder_y = (qword *) dma_y;
	for (int i=1; i<=32; i++){
		qword *qdma_y = (qword *) (dma_y - i*64);
		*qdma_y = *qborder_y;
	}

	uint64_t *dborder_cb = (uint64_t *) dma_cb;
	uint64_t *dborder_cr = (uint64_t *) dma_cr;
	for (int i=1; i<=16; i++){
		uint64_t *ddma_cb = (uint64_t *) (dma_cb - i*32);
		uint64_t *ddma_cr = (uint64_t *) (dma_cr - i*32);
		*ddma_cb = *dborder_cb;
		*ddma_cr = *dborder_cr;
	}
}

static void extend_edges_bottom(uint8_t *dma_y, uint8_t *dma_cb, uint8_t *dma_cr){
	qword *qborder_y = (qword *) dma_y;
	for (int i=1; i<=32; i++){
		qword *qdma_y = (qword *) (dma_y + i*64);
		*qdma_y = *qborder_y;
	}
	
	uint64_t *dborder_cb = (uint64_t *) dma_cb;
	uint64_t *dborder_cr = (uint64_t *) dma_cr;
	for (int i=1; i<=16; i++){
		uint64_t *ddma_cb = (uint64_t *) (dma_cb + i*32);
		uint64_t *ddma_cr = (uint64_t *) (dma_cr + i*32);
		*ddma_cb = *dborder_cb;
		*ddma_cr = *dborder_cr;
	}
}

static void extend_extra_edge_right(uint8_t *dma_y, uint8_t *dma_cb, uint8_t *dma_cr, uint8_t *extra_y, uint8_t *extra_cb, uint8_t *extra_cr, int lines, int lines_c){

	for (int i=0; i<lines; i++){
		memset(extra_y, dma_y[-1], 32);
		dma_y+=64; extra_y+=32;
	}
	
	for (int i=0; i<lines_c; i++){
		memset(extra_cb, dma_cb[-1], 16);
		memset(extra_cr, dma_cr[-1], 16);
		dma_cb+=32; dma_cr+=32;
		extra_cb+=16; extra_cr+=16;
	}
}

static void extend_extra_edge_top(uint8_t *extra_y, uint8_t *extra_cb, uint8_t *extra_cr){
	qword *qborder_y = (qword *) extra_y;
	qword *qborder_y2 = (qword *) (extra_y+16);
	
	for (int i=1; i<=32; i++){
		qword *qextra_y = (qword *) (extra_y-i*32);
		*qextra_y = *qborder_y;
		*(qextra_y+1) = *qborder_y2;
	}
	
	qword *qborder_cb = (qword *) extra_cb;
	qword *qborder_cr = (qword *) extra_cr;
	for (int i=1; i<=16; i++){
		qword *qextra_cb = (qword *) (extra_cb - i*16);
		qword *qextra_cr = (qword *) (extra_cr - i*16);
		*qextra_cb = *qborder_cb;
		*qextra_cr = *qborder_cr;
	}
}

static void extend_extra_edge_bottom(uint8_t *extra_y, uint8_t *extra_cb, uint8_t *extra_cr){
	qword *qborder_y = (qword *) extra_y;
	qword *qborder_y2 = (qword *) (extra_y+16);
	
	for (int i=1; i<=32; i++){
		qword *qextra_y = (qword *) (extra_y+i*32);
		*qextra_y = *qborder_y;
		*(qextra_y+1) = *qborder_y2;
	}
	
	qword *qborder_cb = (qword *) extra_cb;
	qword *qborder_cr = (qword *) extra_cr;
	for (int i=1; i<=16; i++){
		qword *qextra_cb = (qword *) (extra_cb + i*16);
		qword *qextra_cr = (qword *) (extra_cr + i*16);
		*qextra_cb = *qborder_cb;
		*qextra_cr = *qborder_cr;
	}
}

static void extend_edges(H264Context_spu *h, int mb_x, int mb_y){
	H264slice *s = h->s;
	
	uint8_t *dma_y; 
	uint8_t *dma_cb; 
	uint8_t *dma_cr;
	
	uint8_t *extra_y  = extra_edge_y;
	uint8_t *extra_cb = extra_edge_cb;
	uint8_t *extra_cr = extra_edge_cr;
	
	int pos = (mb_x+2) %4;
	if (mb_x == 0){
		if (mb_y ==0){
			extend_edges_left(&dma_y_ls[32*64], &dma_cb_ls[16*32], &dma_cr_ls[16*32], 12, 6);
		}else if (mb_y == s->mb_height -1){
			extend_edges_left(dma_y_ls, dma_cb_ls, dma_cr_ls, 20, 10);
		}else {
			extend_edges_left(dma_y_ls, dma_cb_ls, dma_cr_ls, 16, 8);
		}
	}else if (mb_x == s->mb_width-1){
		dma_y  = &dma_y_ls [(pos+1)*16];
		dma_cb = &dma_cb_ls[(pos+1)*8];
		dma_cr = &dma_cr_ls[(pos+1)*8];
		if (mb_y ==0){
			dma_y   += 32*64;
			dma_cb  += 16*32;
			dma_cr  += 16*32;
			extra_y = extra_edge_y  + 32*32;
			extra_cb= extra_edge_cb + 16*16;
			extra_cr= extra_edge_cr + 16*16;
			
			if (pos==2){
				extend_edges_right(dma_y, dma_cb, dma_cr, 12, 6, 1);
				extend_extra_edge_right(dma_y, dma_cb, dma_cr, extra_y, extra_cb, extra_cr, 12, 6);
			}else if (pos==3){
				extend_extra_edge_right(dma_y, dma_cb, dma_cr, extra_y, extra_cb, extra_cr, 12, 6);
			}else{
				extend_edges_right(dma_y, dma_cb, dma_cr, 12, 6, 2);
			}
		}else if (mb_y == s->mb_height -1){
			if (pos==2){
				extend_edges_right(dma_y, dma_cb, dma_cr, 20, 10, 1);
				extend_extra_edge_right(dma_y, dma_cb, dma_cr, extra_y, extra_cb, extra_cr, 20, 10);
			}else if (pos==3){
				extend_extra_edge_right(dma_y, dma_cb, dma_cr, extra_y, extra_cb, extra_cr, 20, 10);
			}else{
				extend_edges_right(dma_y, dma_cb, dma_cr, 20, 10, 2);
			}				
		}else {
			if (pos==2){
				extend_edges_right(dma_y, dma_cb, dma_cr, 16, 8, 1);
				extend_extra_edge_right(dma_y, dma_cb, dma_cr, extra_y, extra_cb, extra_cr, 16, 8);
			}else if (pos==3){
				extend_extra_edge_right(dma_y, dma_cb, dma_cr, extra_y, extra_cb, extra_cr, 16, 8);
			}else{
				extend_edges_right(dma_y, dma_cb, dma_cr, 16, 8, 1);
			}
		}
	}
		
	if (mb_y == 0){
		dma_y  = &dma_y_ls [32*64];
		dma_cb = &dma_cb_ls[16*32];
		dma_cr = &dma_cr_ls[16*32];
		extra_y = extra_edge_y  + 32*32;
		extra_cb= extra_edge_cb + 16*16;
		extra_cr= extra_edge_cr + 16*16;
		
		if (mb_x ==0){
			extend_edges_top (dma_y + 0*16, dma_cb +0*8, dma_cr + 0*8);
			extend_edges_top (dma_y + 1*16, dma_cb +1*8, dma_cr + 1*8);
			extend_edges_top (dma_y + 2*16, dma_cb +2*8, dma_cr + 2*8);
		}else if (mb_x == s->mb_width -1){
			if (pos==2){
				extend_edges_top (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
				extend_edges_top (dma_y + (pos+1)*16, dma_cb +(pos+1)*8, dma_cr + (pos+1)*8);
				extend_extra_edge_top(extra_y, extra_cb, extra_cr);
			}else if (pos == 3){
				extend_edges_top (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
				extend_extra_edge_top(extra_y, extra_cb, extra_cr);
			}else{
				extend_edges_top (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
				extend_edges_top (dma_y + (pos+1)*16, dma_cb +(pos+1)*8, dma_cr + (pos+1)*8);
				extend_edges_top (dma_y + (pos+2)*16, dma_cb +(pos+2)*8, dma_cr + (pos+2)*8);
			}			
		}else {
			extend_edges_top (dma_y + pos*16, dma_cb + pos*8, dma_cr + pos*8);
		}
	}else if (mb_y == s->mb_height -1){
		dma_y  = &dma_y_ls [19*64];
		dma_cb = &dma_cb_ls[9*32];
		dma_cr = &dma_cr_ls[9*32];
		extra_y = extra_edge_y  + 19*32;
		extra_cb= extra_edge_cb + 9*16;
		extra_cr= extra_edge_cr + 9*16;
		
		if (mb_x ==0){
			extend_edges_bottom (dma_y + 0*16, dma_cb +0*8, dma_cr + 0*8);
			extend_edges_bottom (dma_y + 1*16, dma_cb +1*8, dma_cr + 1*8);
			extend_edges_bottom (dma_y + 2*16, dma_cb +2*8, dma_cr + 2*8);
		}else if (mb_x == s->mb_width -1){
			if (pos==2){
				extend_edges_bottom (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
				extend_edges_bottom (dma_y + (pos+1)*16, dma_cb +(pos+1)*8, dma_cr + (pos+1)*8);
				extend_extra_edge_bottom(extra_y, extra_cb, extra_cr);
			}else if (pos == 3){
				extend_edges_bottom (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
				extend_extra_edge_bottom(extra_y, extra_cb, extra_cr);
			}else{				
				extend_edges_bottom (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
				extend_edges_bottom (dma_y + (pos+1)*16, dma_cb +(pos+1)*8, dma_cr + (pos+1)*8);
				extend_edges_bottom (dma_y + (pos+2)*16, dma_cb +(pos+2)*8, dma_cr + (pos+2)*8);
			}
		}else {
			extend_edges_bottom (dma_y + pos*16, dma_cb +pos*8, dma_cr + pos*8);
		}
	}
}

static void send_pic_data(H264Context_spu *h, int mb_x, int mb_y, int pos, int stride_y, int stride_c){
	H264slice *s = h->s;
	int lines, lines_c;
	int linesize = s->linesize;
	int uvlinesize = s->uvlinesize;
	
	uint8_t* dst_y  = s->dst_y + (mb_x-pos)*16 + (mb_y*16)*linesize;
	uint8_t* dst_cb = s->dst_cb +(mb_x-pos)*8 + (mb_y*8)*uvlinesize;
	uint8_t* dst_cr = s->dst_cr +(mb_x-pos)*8 + (mb_y*8)*uvlinesize;

	if (mb_y == 0){
		dst_y -= 32 *linesize;
		dst_cb-= 16 *uvlinesize;
		dst_cr-= 16 *uvlinesize;
	}else {
		dst_y -= 4 *linesize;
		dst_cb-= 2 *uvlinesize;
		dst_cr-= 2 *uvlinesize;
	}
	
	if (mb_y == 0){
		lines = 12+32; lines_c=6+16;
	}else if (mb_y == s->mb_height-1){
		lines = 20+32; lines_c=10+16;
	}else{
		lines = 16; lines_c=8;
	}
	
	put_list = put_list_buf;
	put_dma_list(dma_y_ls, dst_y, stride_y, lines, linesize, MBD_pic);
	put_dma_list(dma_cb_ls, dst_cb, stride_c, lines_c, uvlinesize, MBD_pic);
	put_dma_list(dma_cr_ls, dst_cr, stride_c, lines_c, uvlinesize, MBD_pic);

	if (mb_x == s->mb_width-1 && pos>1){		
		put_dma_list(extra_edge_y, dst_y+64, 32, lines, linesize, MBD_pic);
		put_dma_list(extra_edge_cb, dst_cb+32, 16, lines_c, uvlinesize, MBD_pic);
		put_dma_list(extra_edge_cr, dst_cr+32, 16, lines_c, uvlinesize, MBD_pic);
   	}
}

void copy_data_and_send(H264Context_spu *h, int mb_x, int mb_y, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c){
	H264slice *s = h->s;
	int lines, lines_c;
	int pos = (mb_x+2)%4; //4 slots in our 64 byte wide transfer buffer. Offset 2 for edge emulation
	uint8_t *dma_y = &dma_y_ls[pos*16];
	uint8_t *dma_cb = &dma_cb_ls[pos*8];
	uint8_t *dma_cr = &dma_cr_ls[pos*8];
	
	if (mb_y == 0){
		dma_y += 32*64;
		dma_cb+= 16*32;
		dma_cr+= 16*32;
	}else{		
		dest_y -= 4*stride_y;
		dest_cb-= 2*stride_c;
		dest_cr-= 2*stride_c;		
	}
	
	if (mb_y == 0){
		lines = 12; lines_c=6;
	}else if (mb_y == s->mb_height-1){
		lines = 20; lines_c=10;
	}else{
		lines = 16; lines_c=8;
	}

	for(int i=0; i<lines; i++){
		qword *qdest_y = (qword *) dest_y;
		qword *qdma_y  = (qword *) dma_y;
		*qdma_y = *qdest_y;
		dma_y +=64;
		dest_y+=stride_y;
	}

	for(int i=0; i<lines_c; i++){
		uint64_t *ddest_cb  = (uint64_t *) dest_cb;
		uint64_t *ddest_cr  = (uint64_t *) dest_cr;
		uint64_t *ddma_cb   = (uint64_t *) dma_cb;
		uint64_t *ddma_cr   = (uint64_t *) dma_cr;
		*ddma_cb = *ddest_cb;
		*ddma_cr = *ddest_cr;
		dma_cb +=32;
		dma_cr +=32;
		dest_cb+=stride_c;
		dest_cr+=stride_c;
	}

	extend_edges(h, mb_x, mb_y);

	//send when dma buf is full
	if (pos==3){
		send_pic_data(h, mb_x, mb_y, pos, 64, 32);
	} else if (mb_x == s->mb_width-1){
		send_pic_data(h, mb_x, mb_y, pos, 64, 32);
	}
}

static void shift_left(int mb_y, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c){
	int lines, lines_c;
	if (mb_y > 0){
		lines  =20;
		lines_c=10;
		dest_y  -= 4*stride_y;
		dest_cb -= 2*stride_c;
		dest_cr -= 2*stride_c;
	}else {
		lines  =16;
		lines_c= 8;		
	}		
		
	for (int i=0; i<lines; i++){
		qword *left_y  = (qword *) (dest_y -16);
		qword *qdest_y = (qword *) dest_y;
		*left_y = *qdest_y;
		dest_y += stride_y;
	}
	
	for (int i=0; i<lines_c; i++){
		uint64_t *left_cb  = (uint64_t *) (dest_cb -8);
		uint64_t *left_cr  = (uint64_t *) (dest_cr -8);
		uint64_t *ddest_cb = (uint64_t *) dest_cb;
		uint64_t *ddest_cr = (uint64_t *) dest_cr;
		*left_cb = *ddest_cb;
		*left_cr = *ddest_cr;
		dest_cb += stride_c;
		dest_cr += stride_c;
	}
}

void hl_decode_mb_internal(H264Context_spu *h, int stride_y, int stride_c){
	H264slice *s = h->s;
	H264Mb *mb = h->mb;
    const int mb_x= mb->mb_x;
    const int mb_y= mb->mb_y;    
    const int mb_type= mb->mb_type;
	
	uint8_t *dest_y, *dest_cb, *dest_cr;	//ls ptrs (abstracts the fact it is operating in a ls buffer)

    int i;
  
    void (*idct_add)(uint8_t *dst, DCTELEM *block, int stride);
    void (*idct_dc_add)(uint8_t *dst, DCTELEM *block, int stride);

	dest_y  = dest_y_ls + 16 + 4*stride_y;
	dest_cb = dest_cb_ls + 8 + 2*stride_c;
	dest_cr = dest_cr_ls + 8 + 2*stride_c;
	
	if(IS_8x8DCT(mb_type)){
		idct_dc_add = ff_idct8_dc_add;
		idct_add = h->dsp.h264_idct_add[0];
	}
	else{
		idct_dc_add = ff_idct_dc_add;
		idct_add = h->dsp.h264_idct_add[1];
	}

	if (mb_y>0){
		copy_top_borders(mb_x, dest_y, dest_cb, dest_cr, stride_y, stride_c);
	}

	if(IS_INTRA(mb_type)){
		xchg_mb_border(h, dest_y, dest_cb, dest_cr, stride_y, stride_c, 1);

		h->hpc.pred8x8[ mb->chroma_pred_mode ](dest_cb, stride_c);
		h->hpc.pred8x8[ mb->chroma_pred_mode ](dest_cr, stride_c);

		if(IS_INTRA4x4(mb_type)){
			if(IS_8x8DCT(mb_type)){

				for(i=0; i<16; i+=4){
					uint8_t * const ptr= dest_y + block_offset[i];
					const int dir= mb->intra4x4_pred_mode_cache[ scan8[i] ];
					const int nnz = mb->non_zero_count_cache[ scan8[i] ];
					h->hpc.pred8x8l[ dir ](ptr, (mb->topleft_samples_available<<i)&0x8000,
												(mb->topright_samples_available<<i)&0x4000, stride_y);

					if(nnz){
						if(nnz == 1 && mb->mb[i*16])
							idct_dc_add(ptr, mb->mb + i*16, stride_y);
						else{
							idct_add   (ptr, mb->mb + i*16, stride_y);
						}
					}
				}
			}else{
				for(i=0; i<16; i++){
					uint8_t * const ptr= dest_y + block_offset[i];
					const int dir= mb->intra4x4_pred_mode_cache[ scan8[i] ];

					uint8_t *topright;
					int nnz, tr;
					if(dir == DIAG_DOWN_LEFT_PRED || dir == VERT_LEFT_PRED){
						const int topright_avail= (mb->topright_samples_available<<i)&0x8000;
						if(!topright_avail){
							tr= ptr[3 - stride_y]*0x01010101;
							topright= (uint8_t*) &tr;
						}else
							topright= ptr + 4 - stride_y;
					}else
						topright= NULL;

					h->hpc.pred4x4[ dir ](ptr, topright, stride_y);
					nnz = mb->non_zero_count_cache[ scan8[i] ];
					if(nnz){
						if(nnz == 1 && mb->mb[i*16])
							idct_dc_add(ptr, mb->mb + i*16, stride_y);
						else
							idct_add   (ptr, mb->mb + i*16, stride_y);
					}
				}
			}

		}else{
			h->hpc.pred16x16[ mb->intra16x16_pred_mode ](dest_y , stride_y);
			h264_luma_dc_dequant_idct_c(mb->mb, mb->dequant4_coeff_y);
		}
		xchg_mb_border(h, dest_y, dest_cb, dest_cr, stride_y, stride_c, 0);

	}else {
		hl_motion(h, dest_y, dest_cb, dest_cr, stride_y, stride_c);
	}

	if(!IS_INTRA4x4(mb_type)){
		if(IS_INTRA16x16(mb_type)){
			for(i=0; i<16; i++){
				if(mb->non_zero_count_cache[ scan8[i] ])
					idct_add(dest_y + block_offset[i], mb->mb + i*16, stride_y);
				else if(mb->mb[i*16])
					idct_dc_add(dest_y + block_offset[i], mb->mb + i*16, stride_y);
			}
		}else if(mb->cbp&15){
			const int incr = IS_8x8DCT(mb_type) ? 4 : 1;
			for(i=0; i<16; i+=incr){
				int nnz = mb->non_zero_count_cache[ scan8[i] ];
				if(nnz){
					if(nnz==1 && mb->mb[i*16])
						idct_dc_add(dest_y + block_offset[i], mb->mb + i*16, stride_y);
					else
						idct_add(dest_y + block_offset[i], mb->mb + i*16, stride_y);
				}
			}
		}
	}

	if(mb->cbp&0x30){
		uint8_t *dest[2] = {dest_cb, dest_cr};
		chroma_dc_dequant_idct_c(mb->mb + 16*16, mb->dequant4_coeff_cb);
		chroma_dc_dequant_idct_c(mb->mb + 16*16+4*16, mb->dequant4_coeff_cr);

		idct_add = h->dsp.h264_idct_add[1];
		idct_dc_add = ff_idct_dc_add;
		for(i=16; i<16+8; i++){
			if(mb->non_zero_count_cache[ scan8[i] ])
				idct_add   (dest[(i&4)>>2] + block_offset[i], mb->mb + i*16, stride_c);
			else if(mb->mb[i*16])
				idct_dc_add(dest[(i&4)>>2] + block_offset[i], mb->mb + i*16, stride_c);
		}
	}

	// save unfiltered borders
	backup_mb_border(h, dest_y, dest_cb, dest_cr, stride_y, stride_c);
	if (mb->deblock_mb){
		filter_mb( h, dest_y, dest_cb, dest_cr, stride_y, stride_c);
	}

	if (mb_y < s->mb_height-1){
		if(mb_x>0){
			send_top_borders(h, mb_x-1, dest_y-16, dest_cb-8, dest_cr-8, stride_y, stride_c);
		}
		if (mb_x == s->mb_width-1){
			send_top_borders(h, mb_x, dest_y, dest_cb, dest_cr, stride_y, stride_c);
		}
	}
	update_tgt_spe_dep(h, 0);

	if (h->blocking){
		if (mb_x>0){			
			copy_data_and_send(h, mb_x-1, mb_y, dest_y-16, dest_cb-8, dest_cr-8, stride_y, stride_c);
			wait_dma_id(MBD_pic);
		}
		if (mb_x == s->mb_width-1){			
			copy_data_and_send(h, mb_x, mb_y, dest_y, dest_cb, dest_cr, stride_y, stride_c);
			wait_dma_id(MBD_pic);
		}
		
	}else{
		if (mb_x>0){
			wait_dma_id(MBD_pic);
			copy_data_and_send(h, mb_x-1, mb_y, dest_y-16, dest_cb-8, dest_cr-8, stride_y, stride_c);
		}
		if (mb_x == s->mb_width-1){
			wait_dma_id(MBD_pic);
			copy_data_and_send(h, mb_x, mb_y, dest_y, dest_cb, dest_cr, stride_y, stride_c);
		}
	}

	if (mb_x < s->mb_width)
		shift_left(mb_y, dest_y, dest_cb, dest_cr, stride_y, stride_c);
	
}
