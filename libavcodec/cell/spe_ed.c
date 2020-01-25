/*
 * Copyright 2013 TU Berlin
 *
 * This file is part of Starbench.
 *
 * This Starbench is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Starbench is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 * @author Chi Ching Chi <chi.c.chi@tu-berlin.de>
 */

#define CELL_SPE

#include <string.h>
#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include "libavcodec/avcodec.h"
#include "h264_cabac_spu.h"
#include "cabac_spu.h"
#include "h264_types_spu.h"
#include "h264_tables.h"
#include "h264_dma.h"
#include "h264_tables.h"

#define MB_WIDTH 240
#define MB_STRIDE (MB_WIDTH+16)

H264Cabac_spu hcabac;
CABACContext cabac;
DECLARE_ALIGNED_16(EDSlice_spu, slice[2]);
DECLARE_ALIGNED_16(H264Mb, mb[2]);
DECLARE_ALIGNED_16(H264spe, spe);

DECLARE_ALIGNED_16(uint8_t, non_zero_count_table[2][MB_STRIDE][32]);
DECLARE_ALIGNED_16(uint8_t, mvd_table[2][2][8*MB_STRIDE][2]);
DECLARE_ALIGNED_16(uint8_t, direct_table[2][4*MB_STRIDE]);
DECLARE_ALIGNED_16(uint8_t, chroma_pred_mode_table[2][MB_STRIDE]);
DECLARE_ALIGNED_16(uint8_t, intra4x4_pred_mode_table[2][8*MB_STRIDE]);
DECLARE_ALIGNED_16(uint16_t,cbp_table[2][MB_STRIDE]);
DECLARE_ALIGNED_16(uint8_t, qscale_table[2][MB_STRIDE]);

DECLARE_ALIGNED_16(uint32_t, mb_type_table[2][MB_STRIDE]);
DECLARE_ALIGNED_16(int8_t, ref_index_table[2][2][4*MB_STRIDE]);
DECLARE_ALIGNED_16(int16_t, motion_val_table[2][2][4*4*MB_WIDTH][2]);

DECLARE_ALIGNED(128, uint8_t, bytestream_ls[4096]);
DECLARE_ALIGNED_16(uint32_t, list1_mb_type_table[2][MB_STRIDE]);
DECLARE_ALIGNED_16(int8_t, list1_ref_index_table[2][2][4*MB_STRIDE]);

DECLARE_ALIGNED_16(spe_pos, dma_temp); //dma temp for sending
//mb position of neighbouring spes
DECLARE_ALIGNED_16(volatile spe_pos, src_spe); //written by SPE_ID -1
static int total_lines;

static inline int dep_resolved(H264spe *p){
	int spe_id = p->spe_id;
	volatile int lines_proc = src_spe.count;
	if (spe_id==0)
		return (total_lines < lines_proc-1 +p->mb_height)? 1:0;
	else
		return (total_lines < lines_proc-1)? 1:0;
}

static void update_tgt_spe_dep(H264spe *p, int end){
	// 	if (end ){
   total_lines++;
   spe_pos* dma_spe = &dma_temp;
   spe_pos* tgt_spe = p->tgt_spe + (unsigned) &src_spe; //located in target spe local store
   dma_spe->count = end? total_lines+1: total_lines;
   spu_dma_barrier_put(dma_spe, (unsigned) tgt_spe, sizeof(dma_temp), ED_put);
   // 	}

}

static int init_cabac(H264spe *p, H264Cabac_spu *hc){
	hc->mb_height = p->mb_height;
	hc->mb_width = p->mb_width;
	hc->b_stride = 4*p->mb_width;
	hc->mb_stride = p->mb_stride;

	for(int i=0; i<16; i++){
		#define T(x) (x>>2) | ((x<<2) & 0xF)
		hc->zigzag_scan[i] = T(zigzag_scan[i]);
		#undef T
	}
	for(int i=0; i<64; i++){
		#define T(x) (x>>3) | ((x&7)<<3)
		hc->zigzag_scan8x8[i] = T(ff_zigzag_direct[i]);
		#undef T
	}
}

static void reset_cabac_buffers(){
 memset(intra4x4_pred_mode_table, 0, sizeof(intra4x4_pred_mode_table));
	memset(mvd_table, 0, sizeof(mvd_table));
	memset(direct_table, 0, sizeof(direct_table));
	memset(chroma_pred_mode_table, 0, sizeof(chroma_pred_mode_table));
	memset(cbp_table, 0, sizeof(cbp_table));
	memset(qscale_table, 0, sizeof(qscale_table));
 	memset(mb_type_table, 0, sizeof(mb_type_table));
	memset(ref_index_table, 0, sizeof(ref_index_table));
	memset(motion_val_table, 0, sizeof(motion_val_table));
}

static void ff_init_cabac_decoder(CABACContext *c, const uint8_t *buf, int bufsize){
	int align = (unsigned) buf & 0xF;
	int dma_size;

	c->bytestream_ea_start=
	c->bytestream_ea= buf;
	c->bytestream_ea_end= buf + bufsize;
	c->bufsize = bufsize;

	if (bufsize + align >= sizeof(bytestream_ls)){
		dma_size = sizeof(bytestream_ls);
		c->bufsize = c->bufsize +align - sizeof(bytestream_ls);
	}else{
		int align_end = (bufsize+align) &0xF;
		if (align_end)
			dma_size = bufsize+align + 16-align_end;
		else
			dma_size = bufsize+align;
		c->bufsize = 0;
	}
// 	printf("%d\n", dma_size);
	c->bytestream_end  = &bytestream_ls[dma_size];
	c->bytestream_start= c->bytestream = &bytestream_ls[align];
 	spu_dma_get(bytestream_ls, (unsigned) buf - align, dma_size, ED_get );
	c->bytestream_ea_start=
	c->bytestream_ea= buf + dma_size -align;

	wait_dma_id(ED_get);

	if (align %2){
		c->low =  (*c->bytestream++)<<18;
		c->low+=  (*c->bytestream++)<<10;
		c->low+= ((*c->bytestream++)<<2) + 2;
	}else {
		c->low =  (*c->bytestream++)<<18;
		c->low+=  (*c->bytestream++)<<10;
		c->low+=  (2<<8);
	}

	c->range= 0x1FE;
	bytecount=0;
}

static void init_dequant8_coeff_table(EDSlice_spu *s, H264Cabac_spu *hc){
    int i,q,x;
    const int transpose = HAVE_ALTIVEC;
    hc->dequant8_coeff[0] = hc->dequant8_buffer[0];
    hc->dequant8_coeff[1] = hc->dequant8_buffer[1];

    for(i=0; i<2; i++){
        if(i && !memcmp(s->pps.scaling_matrix8[0], s->pps.scaling_matrix8[1], 64*sizeof(uint8_t))){
            hc->dequant8_coeff[1] = hc->dequant8_buffer[0];
            break;
        }

        for(q=0; q<52; q++){
            int shift = div6[q];
            int idx = rem6[q];
            for(x=0; x<64; x++)
                hc->dequant8_coeff[i][q][transpose ? (x>>3)|((x&7)<<3) : x] =
                    ((uint32_t)dequant8_coeff_init[idx][ dequant8_coeff_init_scan[((x>>1)&12) | (x&3)] ] *
                    s->pps.scaling_matrix8[i][x]) << shift;
        }
    }
}

static void init_dequant4_coeff_table(EDSlice_spu *s, H264Cabac_spu *hc){
    int i,j,q,x;
    const int transpose = HAVE_MMX | HAVE_ALTIVEC | HAVE_NEON;
    for(i=0; i<6; i++ ){
        hc->dequant4_coeff[i] = hc->dequant4_buffer[i];
        for(j=0; j<i; j++){
            if(!memcmp(s->pps.scaling_matrix4[j], s->pps.scaling_matrix4[i], 16*sizeof(uint8_t))){
                hc->dequant4_coeff[i] = hc->dequant4_buffer[j];
                break;
            }
        }
        if(j<i)
            continue;

        for(q=0; q<52; q++){
            int shift = div6[q] + 2;
            int idx = rem6[q];
            for(x=0; x<16; x++)
                hc->dequant4_coeff[i][q][transpose ? (x>>2)|((x<<2)&0xF) : x] =
                    ((uint32_t)dequant4_coeff_init[idx][(x&1) + ((x>>2)&1)] *
                    s->pps.scaling_matrix4[i][x]) << shift;
        }
    }
}

static void init_dequant_tables(EDSlice_spu *s, H264Cabac_spu *hc){
    int i,x;

    init_dequant4_coeff_table(s, hc);
    if(s->pps.transform_8x8_mode)
        init_dequant8_coeff_table(s, hc);
    if(s->transform_bypass){
        for(i=0; i<6; i++)
            for(x=0; x<16; x++)
                hc->dequant4_coeff[i][0][x] = 1<<6;
        if(s->pps.transform_8x8_mode)
            for(i=0; i<2; i++)
                for(x=0; x<64; x++)
                    hc->dequant8_coeff[i][0][x] = 1<<6;
    }
}

static void init_entropy_buf(H264Cabac_spu *hc, EDSlice_spu *s){
	hc->non_zero_count_top 		= non_zero_count_table[0];
	hc->non_zero_count     		= non_zero_count_table[1];
	hc->mvd_top[0]				= mvd_table[0][0];
	hc->mvd[0]					= mvd_table[0][1];
	hc->mvd_top[1]				= mvd_table[1][0];
	hc->mvd[1]					= mvd_table[1][1];
	hc->direct_top		   		= direct_table[0];
	hc->direct			   		= direct_table[1];
	hc->chroma_pred_mode_top	= chroma_pred_mode_table[0];
	hc->chroma_pred_mode  		= chroma_pred_mode_table[1];
	hc->intra4x4_pred_mode_top	= intra4x4_pred_mode_table[0];
	hc->intra4x4_pred_mode  	= intra4x4_pred_mode_table[1];
	hc->cbp_top			   		= cbp_table[0];
	hc->cbp				   		= cbp_table[1];
	hc->qscale_top			   	= qscale_table[0] +1;
	hc->qscale				   	= qscale_table[1] +1;

	hc->mb_type_top 			= mb_type_table[0]+1;
	hc->mb_type		 			= mb_type_table[1]+1;
	hc->ref_index_top[0]		= ref_index_table[0][0];
	hc->ref_index_top[1]		= ref_index_table[1][0];
	hc->ref_index[0]			= ref_index_table[0][1];
	hc->ref_index[1]			= ref_index_table[1][1];
	hc->motion_val_top[0] 		= motion_val_table[0][0];
	hc->motion_val_top[1] 		= motion_val_table[1][0];
	hc->motion_val[0] 			= motion_val_table[0][1];
	hc->motion_val[1] 			= motion_val_table[1][1];

	int mb_stride = hc->mb_stride;

	if (s->slice_type_nos == FF_B_TYPE){
		while(!dep_resolved(&spe));
		spu_dma_get(list1_mb_type_table[0], (unsigned) (s->list1.mb_type -1), mb_stride*sizeof(uint32_t), ED_get);
		spu_dma_get(list1_ref_index_table[0][0], (unsigned) s->list1.ref_index[0], mb_stride*4*sizeof(int8_t), ED_get);
		spu_dma_get(list1_ref_index_table[0][1], (unsigned) s->list1.ref_index[1], mb_stride*4*sizeof(int8_t), ED_get);
		wait_dma_id(ED_get);
		spu_dma_get(list1_mb_type_table[1], (unsigned) (s->list1.mb_type -1 + mb_stride), mb_stride*sizeof(uint32_t), ED_get);
		spu_dma_get(list1_ref_index_table[1][0], (unsigned) (s->list1.ref_index[0] + 4*mb_stride), mb_stride*4*sizeof(int8_t), ED_get);
		spu_dma_get(list1_ref_index_table[1][1], (unsigned) (s->list1.ref_index[1] + 4*mb_stride), mb_stride*4*sizeof(int8_t), ED_get);
		hc->list1_mb_type = list1_mb_type_table[0]+1;
		hc->list1_ref_index[0] = list1_ref_index_table[0][0];
		hc->list1_ref_index[1] = list1_ref_index_table[0][1];
	}

}

static void update_entropy_buf(H264Cabac_spu *hc, EDSlice_spu *s, int line){
	int mb_stride = hc->mb_stride;
	int mb_width = hc->mb_width;
	int top = (line+1)%2;
	int cur = line%2;
	int bottom = (line+1)%2; //same as top, but to identify prebuffering of next line.

	hc->non_zero_count_top 		= non_zero_count_table[top];
	hc->non_zero_count     		= non_zero_count_table[cur];
	hc->mvd_top[0]				= mvd_table[0][top];
	hc->mvd[0]					= mvd_table[0][cur];
	hc->mvd_top[1]				= mvd_table[1][top];
	hc->mvd[1]					= mvd_table[1][cur];
	hc->direct_top		   		= direct_table[top];
	hc->direct			   		= direct_table[cur];
	hc->chroma_pred_mode_top	= chroma_pred_mode_table[top];
	hc->chroma_pred_mode  		= chroma_pred_mode_table[cur];
	hc->intra4x4_pred_mode_top	= intra4x4_pred_mode_table[top];
	hc->intra4x4_pred_mode  	= intra4x4_pred_mode_table[cur];
	hc->cbp_top			   		= cbp_table[top];
	hc->cbp				   		= cbp_table[cur];
	hc->qscale_top			   	= qscale_table[top] +1;
	hc->qscale				   	= qscale_table[cur] +1;

	hc->mb_type_top 			= mb_type_table[top]+1;
	hc->mb_type		 			= mb_type_table[cur]+1;
	hc->ref_index_top[0]		= ref_index_table[0][top];
	hc->ref_index_top[1]		= ref_index_table[1][top];
	hc->ref_index[0]			= ref_index_table[0][cur];
	hc->ref_index[1]			= ref_index_table[1][cur];
	hc->motion_val_top[0] 		= motion_val_table[0][top];
	hc->motion_val_top[1] 		= motion_val_table[1][top];
	hc->motion_val[0] 			= motion_val_table[0][cur];
	hc->motion_val[1] 			= motion_val_table[1][cur];

	wait_dma_id(ED_put);

	spu_dma_put(mb_type_table[top], (unsigned) (s->pic.mb_type -1 + line*mb_stride), mb_stride*sizeof(uint32_t), ED_put);
	spu_dma_put(ref_index_table[0][top], (unsigned) (s->pic.ref_index[0] + line*4*mb_stride), 4*mb_stride*sizeof(int8_t), ED_put);
	spu_dma_put(ref_index_table[1][top], (unsigned) (s->pic.ref_index[1] + line*4*mb_stride), 4*mb_stride*sizeof(int8_t), ED_put);
	spu_dma_put(motion_val_table[0][top], (unsigned) (s->pic.motion_val[0]+ line*16*mb_width), 16*mb_width*2*sizeof(int16_t), ED_put);
	spu_dma_put(motion_val_table[1][top], (unsigned) (s->pic.motion_val[1]+ line*16*mb_width), 16*mb_width*2*sizeof(int16_t), ED_put);

	if (s->slice_type_nos == FF_B_TYPE){
		update_tgt_spe_dep(&spe, 0);
		wait_dma_id(ED_get);

		if (line + 2 < hc->mb_height){
			while(!dep_resolved(&spe));
			spu_dma_get(list1_mb_type_table[cur], (unsigned) (s->list1.mb_type -1 + (line+2)*mb_stride), mb_stride*sizeof(uint32_t), ED_get);
			spu_dma_get(list1_ref_index_table[cur][0], (unsigned) (s->list1.ref_index[0] + (line+2)*4*mb_stride), mb_stride*4*sizeof(int8_t), ED_get);
			spu_dma_get(list1_ref_index_table[cur][1], (unsigned) (s->list1.ref_index[1] + (line+2)*4*mb_stride), mb_stride*4*sizeof(int8_t), ED_get);
		}
		hc->list1_mb_type = list1_mb_type_table[bottom]+1;
		hc->list1_ref_index[0] = list1_ref_index_table[bottom][0];
		hc->list1_ref_index[1] = list1_ref_index_table[bottom][1];
	}

}

// void printmbdiff(EDSlice_spu *s, H264Cabac_spu *hc, H264Mb *mp, H264Mb *ms){
//
// 	printf("mb_x %d, %d\n", mp->mb_x, ms->mb_x);
// 	printf("mb_y %d, %d\n", mp->mb_y, ms->mb_y);
// 	printf("mb_xy %d, %d\n", mp->mb_xy, ms->mb_xy);
// 	printf("top_mb_xy %d, %d\n", mp->top_mb_xy, ms->top_mb_xy);
// 	printf("left_mb_xy %d, %d\n", mp->left_mb_xy, ms->left_mb_xy);
// 	printf("chroma_pred_mode %d, %d\n", mp->chroma_pred_mode, ms->chroma_pred_mode);
// 	printf("intra16x16_pred_mode %d, %d\n", mp->intra16x16_pred_mode, ms->intra16x16_pred_mode);
// 	printf("topleft_samples %d, %d\n", mp->topleft_samples_available, ms->topleft_samples_available);
// 	printf("topright_samples %d, %d\n", mp->topright_samples_available, ms->topright_samples_available);
// 	printf("top_samples %d, %d\n", mp->top_samples_available, ms->top_samples_available);
// 	printf("left_samples %d, %d\n", mp->left_samples_available, ms->left_samples_available);
//
// 	if (memcmp(mp->intra4x4_pred_mode_cache, ms->intra4x4_pred_mode_cache, 40)){
// 		for (int i=0; i<5; i++){
// 			for (int j=0; j<8; j++){
// 				printf("%d, %d\t", mp->intra4x4_pred_mode_cache[i*8+j],ms->intra4x4_pred_mode_cache[i*8+j]);
// 			}
// 			printf("\n");
// 		}
// 	}
//
// 	if (memcmp(mp->non_zero_count_cache, ms->non_zero_count_cache, 48)){
// 		for (int i=0; i<6; i++){
// 			for (int j=0; j<8; j++){
// 				printf("%u, %u\t", mp->non_zero_count_cache[i*8+j],ms->non_zero_count_cache[i*8+j]);
// 			}
// 			printf("\n");
// 		}
// 	}
//
// 	if (memcmp(mp->sub_mb_type, ms->sub_mb_type, 8)){
// 		for (int i=0; i<4; i++){
// 			printf("%u, %u\t", mp->sub_mb_type[i], mp->sub_mb_type[i]);
// 			printf("\n");
// 		}
// 	}
//
// 	if (memcmp(mp->mv_cache, ms->mv_cache, 320)){
// 		for (int k=0; k<2; k++){
// 			for (int i=0; i<5; i++){
// 				for (int j=0; j<8; j++){
// 					printf("%d, %d, %d, %d\t", mp->mv_cache[k][i*8+j][0], mp->mv_cache[k][i*8+j][1], ms->mv_cache[k][i*8+j][0], ms->mv_cache[k][i*8+j][1]);
// 				}
// 				printf("\n");
// 			}
// 		}
// 	}
//
// 	if (memcmp(mp->ref_cache, ms->ref_cache, 80)){
// 		for (int k=0; k<2; k++){
// 			for (int i=0; i<5; i++){
// 				for (int j=0; j<8; j++){
// 					printf("%d, %d\t", mp->ref_cache[k][i*8+j], ms->ref_cache[k][i*8+j]);
// 				}
// 				printf("\n");
// 			}
// 		}
// 	}
//
// 	printf("cbp %d, %d\n", mp->cbp, ms->cbp);
// 	for (int i=0; i<hc->mb_stride; i++){
//    		printf("%d, ", hc->cbp[i]); fflush(0);
//    	}
// 	printf("\n");
//
// 	printf("mb_type %x, %x\n", mp->mb_type, ms->mb_type);
// 	printf("mb_type IS_INTRA %d, IS_INTRA16x16 %d, IS_DIRECT %d\n", IS_INTRA(ms->mb_type), IS_INTRA16x16(ms->mb_type), IS_DIRECT(ms->mb_type) );
// 	printf("left_type %d, %d\n", mp->left_type, ms->left_type);
// 	printf("top_type %d, %d\n", mp->top_type, ms->top_type);
// 	printf("qscale_mb_xy %d, %d\n", mp->qscale_mb_xy, ms->qscale_mb_xy);
// 	printf("qscale_left_mb_xy %d, %d\n", mp->qscale_left_mb_xy, ms->qscale_left_mb_xy);
// 	printf("qscale_top_mb_xy %d, %d\n", mp->qscale_top_mb_xy, ms->qscale_top_mb_xy);
// // 	for (int i=0; i<hc->mb_stride; i++){
// // 		printf("%d, ", qscale_table[0][i]); fflush(0);
// // 	}
//
// 	if (memcmp(mp->mb, ms->mb, 768)){
// 		for (int i=0; i<16; i++){
// 			for (int j=0; j<16; j++){
// 				printf("%d, %d\t", mp->mb[j + i*16], ms->ref_cache[j + i*16]);
// 			}
// 			printf("\n");
// 		}
// 		for (int i=0; i<8; i++){
// 			for (int j=0; j<8; j++){
// 				printf("%d, %d\t", mp->mb[256 + j + i*8], ms->ref_cache[j + i*8]);
// 			}
// 			printf("\n");
// 		}
// 		for (int i=0; i<8; i++){
// 			for (int j=0; j<8; j++){
// 				printf("%d, %d\t", mp->mb[320+ j + i*8], ms->ref_cache[j + i*8]);
// 			}
// 			printf("\n");
// 		}
// 	}
//
// 	if (memcmp(mp->bS, ms->bS, 32)){
// 		for (int k=0; k<2; k++){
// 			for (int i=0; i<4; i++){
// 				for (int j=0; j<4; j++){
// 					printf("%d, %d\t", mp->bS[k][i][j], mp->mv_cache[k][i][j]);
// 				}
// 				printf("\n");
// 			}
// 		}
// 	}
// 	if (memcmp(mp->edges, ms->edges, 4)){
// 		printf("edges %d, %d, %d, %d\n", mp->edges[0], ms->edges[0], mp->edges[1], ms->edges[1]);
// 		printf("deblock %d, %d\n", mp->deblock_mb, ms->deblock_mb);
// 	}
//
// 	printf("dequant4_coeff_y %d, %d\n", mp->dequant4_coeff_y, ms->dequant4_coeff_y);
// 	printf("dequant4_coeff_cb %d, %d\n", mp->dequant4_coeff_cb, ms->dequant4_coeff_cb);
// 	printf("dequant4_coeff_cr %d, %d\n", mp->dequant4_coeff_cr, ms->dequant4_coeff_cr);
// }
// DECLARE_ALIGNED_16(H264Mb, tmp);


int main(unsigned long long id, unsigned long long argp){
	EDSlice_spu *s;
	H264Cabac_spu *hc = &hcabac;
	CABACContext *c = &cabac;
	H264spe *p = &spe;

	spu_write_out_mbox((unsigned) slice);
	spu_dma_get(p, (unsigned) argp, sizeof(H264spe), ED_spe); //ID_slice is used out of convienience
	wait_dma_id(ED_spe);

	ff_init_cabac_states();
	init_cabac(p, hc);
	hc->blocking=0;
	for(;;){
		spu_read_in_mbox();
		s = &slice[0];
		reset_cabac_buffers();
		init_entropy_buf(hc, s);

		if (hc->blocking) wait_dma_id(ED_get);
		//printf("framesize %d\n", s->byte_bufsize);fflush(0);
 		init_dequant_tables(s, hc);
		ff_init_cabac_decoder( c, s->bytestream_start, s->byte_bufsize );
 		ff_h264_init_cabac_states(s, c);

		int mb_slot=0;
 		for(int j=0; j<hc->mb_height; j++){
			for(int i=0; i<hc->mb_width; i++){
				int eos,ret;
				H264Mb *m = &mb[mb_slot];
				m->mb_x=i;
				m->mb_y=j;
				s->m = m;

				ret = ff_h264_decode_mb_cabac(hc, s, c);

// 				spu_dma_get(&tmp, (unsigned) &s->mbs[j*hc->mb_width + i], sizeof(H264Mb), ED_get);
// 				wait_dma_id(ED_get);
// 				if (memcmp(&tmp, m, sizeof(H264Mb))){
// 					printf("coded pic num %d\n", s->coded_pic_num);
// 					printmbdiff(s, hc,&tmp, m);
// 					return 0;
// 				}
				//printf("qscale %d\n", m->qscale_mb_xy);
				if (!hc->blocking){
					if (mb_slot){
						spu_dma_put(m, (unsigned) &s->mbs[j*hc->mb_width + i], sizeof(H264Mb), ED_putmb1);
						wait_dma_id(ED_putmb0);
					}else {
						spu_dma_put(m, (unsigned) &s->mbs[j*hc->mb_width + i], sizeof(H264Mb), ED_putmb0);
						wait_dma_id(ED_putmb1);
					}
					mb_slot++; mb_slot%=2;
				}else {
					spu_dma_put(m, (unsigned) &s->mbs[j*hc->mb_width + i], sizeof(H264Mb), ED_putmb0);
					wait_dma_id(ED_putmb0);
				}


				eos = get_cabac_terminate( c);

				if( ret < 0) {
					fprintf(stderr, "error at %d bytecount\n", bytecount);
					return -1;
				}
			}
			update_entropy_buf(hc, s, j);
			if (hc->blocking){ wait_dma_id(ED_get); wait_dma_id(ED_put);}
		}
		wait_dma_id(ED_put);
		spu_write_out_mbox(1);

	}

	return 0;


}
