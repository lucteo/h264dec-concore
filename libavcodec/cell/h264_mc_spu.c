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
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <assert.h>

#include "h264_mc_spu.h"
#include "h264_dma.h"
#include "h264_tables.h"
#include "h264_decode_mb_spu.h"


//biweight buffer 
DECLARE_ALIGNED_16(uint8_t, tmp_y_ls[48*16]);	      		
DECLARE_ALIGNED_16(uint8_t, tmp_cb_ls[32*8]);
DECLARE_ALIGNED_16(uint8_t, tmp_cr_ls[32*8]);

//ref buffer (double buffered)
DECLARE_ALIGNED_16(uint8_t, mc_ref[2][16*(4+5)*48 + 2*16*(2+1)*32]);
uint8_t* ref_ptr;

/** Motion Compensation functions*/

static void fill_mc_part(H264mc *mc, int n, int chroma_height, int x_offset, int y_offset, int itp, int weight, int list0, int list1){
	H264mc_part *mc_part = mc->mc_part + mc->npart;
	mc_part->n =n;
	mc_part->chroma_height =chroma_height;
	mc_part->x_offset = x_offset;
	mc_part->y_offset = y_offset;
	mc_part->itp = itp;
	mc_part->weight = weight;
	mc_part->list0 = list0;
	mc_part->list1 = list1;
	
	mc->npart++;
}

void calc_mc_params(H264Mb* mb, H264mc *mc){
	int mb_type = mb->mb_type;
	mc->npart=0;	

	assert(!IS_INTRA(mb_type));
	if(IS_16X16(mb_type)){
		fill_mc_part(mc, 0, 8, 0, 0, 0, 0, IS_DIR(mb_type, 0, 0), IS_DIR(mb_type, 0, 1));
    }else if(IS_16X8(mb_type)){
		fill_mc_part(mc, 0, 4, 0, 0, 0, 0, IS_DIR(mb_type, 0, 0), IS_DIR(mb_type, 0, 1));
		fill_mc_part(mc, 8, 4, 0, 4, 0, 1, IS_DIR(mb_type, 1, 0), IS_DIR(mb_type, 1, 1));
    }else if(IS_8X16(mb_type)){
		fill_mc_part(mc, 0, 8, 0, 0, 1, 2, IS_DIR(mb_type, 0, 0), IS_DIR(mb_type, 0, 1));
		fill_mc_part(mc, 4, 8, 4, 0, 1, 2, IS_DIR(mb_type, 1, 0), IS_DIR(mb_type, 1, 1));
    }else{
        int i;
        assert(IS_8X8(mb_type));

        for(i=0; i<4; i++){
            const int sub_mb_type= mb->sub_mb_type[i];
            const int n= 4*i;
            int x_offset= (i&1)<<2;
            int y_offset= (i&2)<<1;

			if(IS_SUB_8X8(sub_mb_type)){
				fill_mc_part(mc, n, 4, x_offset, y_offset, 1, 3, IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
            }else if(IS_SUB_8X4(sub_mb_type)){
				fill_mc_part(mc, n, 2, x_offset, y_offset, 1, 4, IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
				fill_mc_part(mc, n+2, 2, x_offset, y_offset+2, 1, 4, IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
            }else if(IS_SUB_4X8(sub_mb_type)){
				fill_mc_part(mc, n, 4, x_offset, y_offset, 2, 5, IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
				fill_mc_part(mc, n+1, 4, x_offset+2, y_offset, 2, 5, IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
            }else{
                int j;
                assert(IS_SUB_4X4(sub_mb_type));
                for(j=0; j<4; j++){
                    int sub_x_offset= x_offset + 2*(j&1);
                    int sub_y_offset= y_offset +   (j&2);
					fill_mc_part(mc, n+j, 2, sub_x_offset, sub_y_offset, 2, 6, IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
                }
            }
        }
    }
}

/**
*	Returns a pointer to mc_buf 
*/
static void* alloc_mc_buf(int size){
	void* ptr = ref_ptr;
	ref_ptr += size;
	return ptr;
}

#define TAG_OFFSET_MC MBD_mc_buf1
static uint8_t* get_mc_data(uint8_t* src_ea, int pic_xoffset, int pic_yoffset, int blk_h, int stride, int linesize, int idx){
	assert(src_ea);
	int unalign;
	unsigned address_align;
	
	uint8_t* ea;
	uint8_t* ref_ptr = alloc_mc_buf(blk_h*stride);

	ea = src_ea + pic_xoffset + pic_yoffset*linesize; 
	address_align = ((unsigned) ea) & 0xFFFFFFF0;
	unalign = ((unsigned) ea) & 0xF;
	get_dma_list(ref_ptr, (void *)address_align, stride, blk_h, linesize, idx + TAG_OFFSET_MC, 0);
	return (ref_ptr + unalign);
}

static uint8_t* get_mc_data_blocking(uint8_t* src_ea, int pic_xoffset, int pic_yoffset, int blk_h, int stride, int linesize, int idx){
	assert(src_ea);
	int unalign;
	unsigned address_align;

	uint8_t* ea;
	uint8_t* ref_ptr = alloc_mc_buf(blk_h*stride);

	ea = src_ea + pic_xoffset + pic_yoffset*linesize;
	address_align = ((unsigned) ea) & 0xFFFFFFF0;
	unalign = ((unsigned) ea) & 0xF;
	get_dma_list(ref_ptr, (void *)address_align, stride, blk_h, linesize, MBD_mc_buf1, 0);
	wait_dma_id(MBD_mc_buf1);
	return (ref_ptr + unalign);
}

//#undef TAG_OFFSET_MC

static void get_mc_components(H264Context_spu *h, H264Mb *mb, H264mc_part* mc_part, Picture_spu *pic, int n, int chroma_height, int list, int src_x_offset, int src_y_offset, int idx){
	assert(pic);
	H264slice *s = h->s;
	ref_data *ref = &mc_part->ref[list];
    const int mx= mb->mv_cache[list][ scan8[n] ][0] + src_x_offset*8;
    const int my= mb->mv_cache[list][ scan8[n] ][1] + src_y_offset*8;

    const int pic_width  = 16*s->mb_width;
    const int pic_height = 16*s->mb_height;
	
	int blk_h= chroma_height*2+5;
	//int blk_w= 8*2+5;
	
	int blk_h_c= chroma_height+1;
	//int blk_w_c= 9;

	int ymx= mx>>2;
    int ymy= my>>2;
    int cmy= my>>3;
    int cmx= mx>>3;

    //truncate the motion vectors references
    if(ymy>= pic_height+2){
        ymy=pic_height+1;
    }else if(ymy <=-19){
        ymy=-18;
    }
    if(ymx>= pic_width+2){
        ymx= pic_width+1;
    }else if(ymx<=-19){
        ymx=-19;
    }

	if(cmy >= pic_height>>1){
        cmy = (pic_height>>1) -1;
    }else if(cmy<=-9){
        cmy=-8;
    }
    if(cmx >= pic_width>>1){
        cmx = (pic_width>>1) -1;
    }else if(cmx<=-9){
        cmx=-8;
    }
	if (!h->blocking){
		ref->data[0]=get_mc_data(pic->data[0], ymx-2, ymy-2, blk_h, STRIDE_Y, s->linesize, idx);
		ref->data[1]=get_mc_data(pic->data[1], cmx, cmy, blk_h_c, STRIDE_C, s->uvlinesize, idx);
		ref->data[2]=get_mc_data(pic->data[2], cmx, cmy, blk_h_c, STRIDE_C, s->uvlinesize, idx);
	} else {
		ref->data[0]=get_mc_data_blocking(pic->data[0], ymx-2, ymy-2, blk_h, STRIDE_Y, s->linesize, idx);
		ref->data[1]=get_mc_data_blocking(pic->data[1], cmx, cmy, blk_h_c, STRIDE_C, s->uvlinesize, idx);
		ref->data[2]=get_mc_data_blocking(pic->data[2], cmx, cmy, blk_h_c, STRIDE_C, s->uvlinesize, idx);

	}
	
}

static void get_ref_data(H264Context_spu *h, H264Mb *mb, H264mc_part *mc_part, int idx){
	H264slice *s = h->s;
	int x_offset = mc_part->x_offset;
	int y_offset = mc_part->y_offset;
	int list0 = mc_part->list0;
	int list1 = mc_part->list1;
	int n = mc_part->n;
	int chroma_height = mc_part->chroma_height;
	Picture_spu *refpic;
	
	x_offset += 8*mb->mb_x;
    y_offset += 8*mb->mb_y;
	
	if(list0){
		refpic= &s->ref_list[0][ mb->ref_cache[0][ scan8[n] ] ];
		get_mc_components(h, mb, mc_part, refpic, n, chroma_height, 0, x_offset, y_offset, idx);
	}
	if(list1){
		refpic= &s->ref_list[1][ mb->ref_cache[1][ scan8[n] ] ];
		get_mc_components(h, mb, mc_part, refpic, n, chroma_height, 1, x_offset, y_offset, idx);
	}
}

void fill_ref_buf(H264Context_spu *h, H264Mb *mb, H264mc *mc){
	int idx = h->mc_idx;
	int i;

	get_list = get_list_buf;
	ref_ptr = mc_ref[idx];
	for(i=0; i<mc->npart; i++){
		get_ref_data(h, mb, &mc->mc_part[i], idx);
	}
}

static void mc_dir_part(H264Context_spu *h, H264mc_part* mc_part, int n, int chroma_height, int list, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, qpel_mc_func *qpix_op, h264_chroma_mc_func chroma_op, int stride_y, int stride_c){
	
	H264Mb *mb = h->mb;
	ref_data* ref = &mc_part->ref[list];
    const int mx= mb->mv_cache[list][ scan8[n] ][0];	//to determine the interpolation mode
    const int my= mb->mv_cache[list][ scan8[n] ][1];
    const int luma_xy= (mx&3) + ((my&3)<<2);
	uint8_t *src_y, *src_cb, *src_cr;
    
	src_y = ref->data[0] +2+2*STRIDE_Y;
	src_cb = ref->data[1];
	src_cr = ref->data[2];
	
	qpix_op[luma_xy](dest_y, src_y, stride_y, chroma_height*2);
	chroma_op(dest_cb, src_cb, stride_c, chroma_height, mx&7, my&7);
	chroma_op(dest_cr, src_cr, stride_c, chroma_height, mx&7, my&7);
}


static void mc_part_biweighted(H264Context_spu *h, H264mc_part *mc_part, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c, h264_biweight_func luma_weight_avg, h264_biweight_func chroma_weight_avg){

	H264Mb *mb = h->mb;
	H264slice *s = h->s;
	int n = mc_part->n;
	int chroma_height = mc_part->chroma_height;
	int itp = mc_part->itp;
	int refn0 = mb->ref_cache[0][ scan8[n] ];
	int refn1 = mb->ref_cache[1][ scan8[n] ];        
	qpel_mc_func *qpix_put=  h->dsp.put_h264_qpel_pixels_tab[itp];
    h264_chroma_mc_func chroma_put= h->dsp.put_h264_chroma_pixels_tab[itp];
    
	// don't optimize for luma-only case, since B-frames usually
	// use implicit weights => chroma too. 
	mc_dir_part(h, mc_part, n, chroma_height, 0, dest_y, dest_cb, dest_cr, qpix_put, chroma_put, stride_y, stride_c);
	
	mc_dir_part(h, mc_part, n, chroma_height, 1, tmp_y_ls, tmp_cb_ls, tmp_cr_ls, qpix_put, chroma_put, STRIDE_Y, STRIDE_C);

	if(s->use_weight == 2){
		int weight0 = s->implicit_weight[refn0][refn1][mb->mb_y&1];
		int weight1 = 64 - weight0;
		luma_weight_avg(  dest_y,  tmp_y_ls, stride_y, STRIDE_Y, 5, weight0, weight1, 0);
		chroma_weight_avg(dest_cb, tmp_cb_ls, stride_c, STRIDE_C, 5, weight0, weight1, 0);
		chroma_weight_avg(dest_cr, tmp_cr_ls, stride_c, STRIDE_C, 5, weight0, weight1, 0);
	}else{
		luma_weight_avg(dest_y, tmp_y_ls, stride_y, STRIDE_Y, s->luma_log2_weight_denom,  s->luma_weight[refn0][0][0] , s->luma_weight[refn1][1][0], s->luma_weight[refn0][0][1] + s->luma_weight[refn1][1][1]);
		
		chroma_weight_avg(dest_cb, tmp_cb_ls, stride_c, STRIDE_C, s->chroma_log2_weight_denom, s->chroma_weight[refn0][0][0][0] , s->chroma_weight[refn1][1][0][0], s->chroma_weight[refn0][0][0][1] + s->chroma_weight[refn1][1][0][1]);
		
		chroma_weight_avg(dest_cr, tmp_cr_ls, stride_c, STRIDE_C, s->chroma_log2_weight_denom, s->chroma_weight[refn0][0][1][0] , s->chroma_weight[refn1][1][1][0], s->chroma_weight[refn0][0][1][1] + s->chroma_weight[refn1][1][1][1]);
	}
}

static void mc_part_weighted(H264Context_spu *h, H264mc_part *mc_part, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c, h264_weight_func luma_weight_op, h264_weight_func chroma_weight_op, int list1){

	H264Mb *mb = h->mb;
	H264slice *s = h->s;

	int n = mc_part->n;
	int chroma_height = mc_part->chroma_height;
	int itp = mc_part->itp;
	qpel_mc_func *qpix_put=  h->dsp.put_h264_qpel_pixels_tab[itp];
    h264_chroma_mc_func chroma_put= h->dsp.put_h264_chroma_pixels_tab[itp];
    
    int list = list1 ? 1 : 0;
	int refn = mb->ref_cache[list][ scan8[n] ];      

	mc_dir_part(h, mc_part, n, chroma_height, list, dest_y, dest_cb, dest_cr, qpix_put, chroma_put, stride_y, stride_c);

	luma_weight_op(dest_y, stride_y, s->luma_log2_weight_denom, s->luma_weight[refn][list][0], s->luma_weight[refn][list][1]);
	if(s->use_weight_chroma){
		chroma_weight_op(dest_cb, stride_c, s->chroma_log2_weight_denom, s->chroma_weight[refn][list][0][0], s->chroma_weight[refn][list][0][1]);
		
		chroma_weight_op(dest_cr, stride_c, s->chroma_log2_weight_denom, s->chroma_weight[refn][list][1][0], s->chroma_weight[refn][list][1][1]);
	}
}


static void mc_part_std(H264Context_spu *h, H264mc_part *mc_part, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c, int list0, int list1){
	int n = mc_part->n;
	int chroma_height = mc_part->chroma_height;
	int itp = mc_part->itp;

    qpel_mc_func *qpix_op=  h->dsp.put_h264_qpel_pixels_tab[itp];
    h264_chroma_mc_func chroma_op= h->dsp.put_h264_chroma_pixels_tab[itp];
    
    if(list0){
        mc_dir_part(h, mc_part, n, chroma_height, 0, dest_y, dest_cb, dest_cr, qpix_op, chroma_op, stride_y, stride_c);

        qpix_op=  h->dsp.avg_h264_qpel_pixels_tab[itp];
        chroma_op= h->dsp.avg_h264_chroma_pixels_tab[itp];
    }

    if(list1){
        mc_dir_part(h, mc_part, n, chroma_height, 1, dest_y, dest_cb, dest_cr, qpix_op, chroma_op, stride_y, stride_c);
    }
}

static void mc_part(H264Context_spu *h, H264mc_part *mc_part, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c){
	H264slice *s = h->s;
	
	int weight = mc_part->weight;
	
	int x_offset = mc_part->x_offset;
	int y_offset = mc_part->y_offset;
	int list0 = mc_part->list0;
	int list1 = mc_part->list1;
    
	dest_y  += 2*x_offset + 2*y_offset*stride_y;
    dest_cb +=   x_offset +   y_offset*stride_c;
    dest_cr +=   x_offset +   y_offset*stride_c;
    
	if(list0 && list1 && s->use_weight !=0){
		h264_biweight_func *weight_avg = &h->dsp.biweight_h264_pixels_tab[weight];
        mc_part_biweighted(h, mc_part, dest_y, dest_cb, dest_cr, stride_y, stride_c, weight_avg[0], weight_avg[3]);
	}
	else if ((list0 || list1) && s->use_weight ==1){
		h264_weight_func *weight_op = &h->dsp.weight_h264_pixels_tab[weight];
		mc_part_weighted(h, mc_part, dest_y, dest_cb, dest_cr, stride_y, stride_c, weight_op[0], weight_op[3], list1);
	}
	else{
        mc_part_std(h, mc_part, dest_y, dest_cb, dest_cr, stride_y, stride_c, list0, list1);
	}
}

void hl_motion(H264Context_spu *h, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr, int stride_y, int stride_c){
	int i;
	H264mc *mc =h->mc; 
	for(i=0; i<mc->npart; i++){
		mc_part(h, &mc->mc_part[i], dest_y, dest_cb, dest_cr, stride_y, stride_c);
	}
}
