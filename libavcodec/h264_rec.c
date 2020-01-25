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


#include "config.h"

#include "dsputil.h"
#include "h264_types.h"
#include "h264_data.h"
#include "h264_mc.h"
#include "h264_deblock.h"
#include "h264_pred_mode.h"
//#undef NDEBUG
#include <assert.h>

void init_mbrec_context(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, int line){
    DecodedPicture *pic = s->curr_pic;
    int mb_stride = mrc->mb_stride;
    int mb_width = mrc->mb_width;
    mrs->mb_type_top = pic->mb_type + (line -1)*mb_stride;
    mrs->mb_type = pic->mb_type + line*mb_stride;
    mrs->ref_index_top[0] = pic->ref_index[0] + 4*(line -1)*mb_stride;
    mrs->ref_index_top[1] = pic->ref_index[1] + 4*(line -1)*mb_stride;
    mrs->ref_index[0] = pic->ref_index[0] + 4*line*mb_stride;
    mrs->ref_index[1] = pic->ref_index[1] + 4*line*mb_stride;

    mrs->motion_val_top[0] = pic->motion_val[0] + 4*mb_width*4*(line-1);
    mrs->motion_val_top[1] = pic->motion_val[1] + 4*mb_width*4*(line-1);
    mrs->motion_val[0] = pic->motion_val[0] + 4*mb_width*4*line;
    mrs->motion_val[1] = pic->motion_val[1] + 4*mb_width*4*line;

    mrs->intra4x4_pred_mode_top = pic->intra4x4_pred_mode + 4*mb_width*(line-1);
    mrs->intra4x4_pred_mode = pic->intra4x4_pred_mode + 4*mb_width*line;

    mrs->non_zero_count_top = pic->non_zero_count + 8*mb_width*(line-1);
    mrs->non_zero_count = pic->non_zero_count + 8*mb_width*line;

    if (s->slice_type_nos == FF_B_TYPE){
        mrs->list1_mb_type = s->dp_ref_list[1][0]->mb_type + line*mb_stride;
        mrs->list1_ref_index[0]  = s->dp_ref_list[1][0]->ref_index[0] + 4*line*mb_stride;
        mrs->list1_ref_index[1]  = s->dp_ref_list[1][0]->ref_index[1] + 4*line*mb_stride;
        mrs->list1_motion_val[0] = s->dp_ref_list[1][0]->motion_val[0] + 4*mb_width*4*line;
        mrs->list1_motion_val[1] = s->dp_ref_list[1][0]->motion_val[1] + 4*mb_width*4*line;
    }

}

#if OMPSS
static void backup_mb_border(H264Mb *m, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize){
    int i;
    uint8_t * top_border_y1 = m->top_border;
    uint8_t * top_border_y2 = m->top_border + 8;
    uint8_t * top_border_cb = m->top_border + 16;
    uint8_t * top_border_cr = m->top_border + 24;
    uint8_t * top_border_next = m->top_border_next;

    src_y  -=   linesize;
    src_cb -= uvlinesize;
    src_cr -= uvlinesize;

    m->left_border[0]= m->top_border[15];
    for(i=1; i<17 ; i++){
        m->left_border[i]= src_y[15 + i*linesize];
    }

    *(uint64_t*)(top_border_y1)   = *(uint64_t*)(src_y +  16*linesize);
    *(uint64_t*)(top_border_next) = *(uint64_t*)(src_y +  16*linesize);
    *(uint64_t*)(top_border_y2)   = *(uint64_t*)(src_y +8+16*linesize);

    m->left_border[17]= m->top_border[16+7];
    m->left_border[17+9]= m->top_border[24+7];
    for(i=1; i<9; i++){
        m->left_border[17  +i]= src_cb[7+i*uvlinesize];
        m->left_border[17+9+i]= src_cr[7+i*uvlinesize];
    }
    *(uint64_t*)(top_border_cb)= *(uint64_t*)(src_cb+8*uvlinesize);
    *(uint64_t*)(top_border_cr)= *(uint64_t*)(src_cr+8*uvlinesize);
}

static void xchg_mb_border(H264Mb *m, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize, int xchg){
    int temp8, i;
    uint64_t temp64;

    uint8_t * top_border_y1 = m->top_border;
    uint8_t * top_border_y2 = m->top_border + 8;
    uint8_t * top_border_cb = m->top_border + 16;
    uint8_t * top_border_cr = m->top_border + 24;
    uint8_t * top_border_next = m->top_border_next;

    int deblock_left;
    int deblock_top;

    deblock_left = (m->mb_x > 0);
    deblock_top =  (m->mb_y > 0);

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
            XCHG(m->left_border[i], src_y [i*  linesize], temp8, xchg);
        }
        XCHG(m->left_border[i], src_y [i*  linesize], temp8, 1);

        for(i = !deblock_top; i<8; i++){
            XCHG(m->left_border[17  +i], src_cb[i*uvlinesize], temp8, xchg);
            XCHG(m->left_border[17+9+i], src_cr[i*uvlinesize], temp8, xchg);
        }
        XCHG(m->left_border[17  +i], src_cb[i*uvlinesize], temp8, 1);
        XCHG(m->left_border[17+9+i], src_cr[i*uvlinesize], temp8, 1);
    }

    if(deblock_top){
        XCHG(*(uint64_t*)(top_border_y1)  , *(uint64_t*)(src_y +1), temp64, xchg);
        XCHG(*(uint64_t*)(top_border_y2)  , *(uint64_t*)(src_y +9), temp64, 1);
        XCHG(*(uint64_t*)(top_border_next), *(uint64_t*)(src_y +17), temp64, 1);

        XCHG(*(uint64_t*)(top_border_cb)  , *(uint64_t*)(src_cb+1), temp64, 1);
        XCHG(*(uint64_t*)(top_border_cr)  , *(uint64_t*)(src_cr+1), temp64, 1);
    }
}
#else

static void backup_mb_border(MBRecContext *d, H264Mb *m, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize){
    int i;
    uint8_t* top_border_y = d->top[m->mb_x].unfiltered_y;
    uint8_t* top_border_cb = d->top[m->mb_x].unfiltered_cb;
    uint8_t* top_border_cr = d->top[m->mb_x].unfiltered_cr;

    uint8_t* left_border_y = d->left.unfiltered_y;
    uint8_t* left_border_cb = d->left.unfiltered_cb;
    uint8_t* left_border_cr = d->left.unfiltered_cr;

    src_y  -=   linesize;
    src_cb -= uvlinesize;
    src_cr -= uvlinesize;

    // There are two lines saved, the line above the top macroblock of a pair,
    // and the line above the bottom macroblock
    left_border_y[0] = top_border_y[15];
    for(i=1; i<17; i++){
        left_border_y[i] = src_y[15+i*  linesize];
    }
    *(uint64_t*)(top_border_y   )   = *(uint64_t*)(src_y +  16*linesize);
    *(uint64_t*)(top_border_y +8)   = *(uint64_t*)(src_y +8+16*linesize);

    left_border_cb[0] = top_border_cb[7];
    left_border_cr[0] = top_border_cr[7];
    for(i=1; i<9; i++){
        left_border_cb[i] = src_cb[7+i*uvlinesize];
        left_border_cr[i] = src_cr[7+i*uvlinesize];
    }
    *(uint64_t*)(top_border_cb)= *(uint64_t*)(src_cb+8*uvlinesize);
    *(uint64_t*)(top_border_cr)= *(uint64_t*)(src_cr+8*uvlinesize);
}

static void xchg_mb_border(MBRecContext *d, H264Mb *m, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize, int xchg){

    int temp8, i;
    uint64_t temp64;
    int deblock_left;
    int deblock_top;

    uint8_t* top_border_y = d->top[m->mb_x].unfiltered_y;
    uint8_t* top_border_cb = d->top[m->mb_x].unfiltered_cb;
    uint8_t* top_border_cr = d->top[m->mb_x].unfiltered_cr;
    uint8_t* top_border_y_next = d->top[m->mb_x +1].unfiltered_y;

    uint8_t* left_border_y = d->left.unfiltered_y;
    uint8_t* left_border_cb = d->left.unfiltered_cb;
    uint8_t* left_border_cr = d->left.unfiltered_cr;

    deblock_left = (m->mb_x > 0);
    deblock_top =  (m->mb_y > 0);

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
        if(m->mb_x+1 < d->mb_width){
            XCHG(*(uint64_t*)(top_border_y_next), *(uint64_t*)(src_y +17), temp64, 1);
        }
        XCHG(*(uint64_t*)(top_border_cb), *(uint64_t*)(src_cb+1), temp64, 1);
        XCHG(*(uint64_t*)(top_border_cr), *(uint64_t*)(src_cr+1), temp64, 1);
    }
}

#endif

void h264_decode_mb_internal(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m){
    int i;
    const int mb_x= m->mb_x;
    const int mb_y= m->mb_y;
    int *block_offset = d->block_offset;

    void (*idct_add)(uint8_t *dst, DCTELEM *block, int stride);
    void (*idct_dc_add)(uint8_t *dst, DCTELEM *block, int stride);

    int linesize   = d->linesize;
    int uvlinesize = d->uvlinesize;

    uint8_t *dest_y  = s->curr_pic->data[0] + (mb_x + mb_y * linesize  ) * 16;
    uint8_t *dest_cb = s->curr_pic->data[1] + (mb_x + mb_y * uvlinesize) * 8;
    uint8_t *dest_cr = s->curr_pic->data[2] + (mb_x + mb_y * uvlinesize) * 8;

    pred_motion_mb_rec (d, mrs, s, m);

    const int mb_type= m->mb_type;

    d->dsp.prefetch(dest_y + (m->mb_x&3)*4*linesize + 64, d->linesize, 4);
    d->dsp.prefetch(dest_cb + (m->mb_x&7)*uvlinesize + 64, dest_cr - dest_cb, 2);

    if(IS_INTRA(mb_type)){
#if OMPSS
        xchg_mb_border(m, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 1);
#else
        xchg_mb_border(d, m, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 1);
#endif

        d->hpc.pred8x8[ m->chroma_pred_mode ](dest_cb, uvlinesize);
        d->hpc.pred8x8[ m->chroma_pred_mode ](dest_cr, uvlinesize);

        if(IS_INTRA4x4(mb_type)){
            if(IS_8x8DCT(mb_type)){
                idct_dc_add = d->hdsp.h264_idct8_dc_add;
                idct_add    = d->hdsp.h264_idct8_add;

                for(i=0; i<16; i+=4){
                    uint8_t * const ptr= dest_y + block_offset[i];
                    const int dir= mrs->intra4x4_pred_mode_cache[ scan8[i] ];

                    const int nnz = mrs->non_zero_count_cache[ scan8[i] ];
                    d->hpc.pred8x8l[ dir ](ptr, (mrs->topleft_samples_available<<i)&0x8000,
                                                (mrs->topright_samples_available<<i)&0x4000, linesize);
                    if(nnz){
                        if(nnz == 1 && m->mb[i*16])
                            idct_dc_add(ptr, m->mb + i*16, linesize);
                        else
                            idct_add   (ptr, m->mb + i*16, linesize);
                    }
                }
            }else{
                idct_dc_add = d->hdsp.h264_idct_dc_add;
                idct_add    = d->hdsp.h264_idct_add;

                for(i=0; i<16; i++){
                    uint8_t * const ptr= dest_y + block_offset[i];
                    const int dir= mrs->intra4x4_pred_mode_cache[ scan8[i] ];
                    uint8_t *topright;
                    int nnz, tr;
                    if(dir == DIAG_DOWN_LEFT_PRED || dir == VERT_LEFT_PRED){
                        const int topright_avail= (mrs->topright_samples_available<<i)&0x8000;
                        assert(mb_y || linesize <= block_offset[i]);
                        if(!topright_avail){
                            tr= ptr[3 - linesize]*0x01010101;
                            topright= (uint8_t*) &tr;
                        }else
                            topright= ptr + 4 - linesize;
                    }else
                        topright= NULL;

                    d->hpc.pred4x4[ dir ](ptr, topright, linesize);
                    nnz = mrs->non_zero_count_cache[ scan8[i] ];
                    if(nnz){
                        if(nnz == 1 && m->mb[i*16])
                            idct_dc_add(ptr, m->mb + i*16, linesize);
                        else
                            idct_add   (ptr, m->mb + i*16, linesize);
                    }
                }
            }
        }else{
            d->hpc.pred16x16[ m->intra16x16_pred_mode ](dest_y , linesize);
        }
#if OMPSS
        xchg_mb_border(m, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 0);
#else
        xchg_mb_border(d, m, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 0);
#endif
    }else {
        hl_motion(d, mrs, s, m, dest_y, dest_cb, dest_cr,
                    d->hdsp.qpel_put, d->dsp.put_h264_chroma_pixels_tab,
                    d->hdsp.qpel_avg, d->dsp.avg_h264_chroma_pixels_tab,
                    d->hdsp.weight_h264_pixels_tab, d->hdsp.biweight_h264_pixels_tab);
    }

    if(!IS_INTRA4x4(mb_type)){

        if(IS_INTRA16x16(mb_type)){

            d->hdsp.h264_idct_add16intra(dest_y, block_offset, m->mb, linesize, mrs->non_zero_count_cache);

        }else if(m->cbp&15){

            if(IS_8x8DCT(mb_type)){
                d->hdsp.h264_idct8_add4(dest_y, block_offset, m->mb, linesize, mrs->non_zero_count_cache);
            }else{
                d->hdsp.h264_idct_add16(dest_y, block_offset, m->mb, linesize, mrs->non_zero_count_cache);
            }
        }
    }

    if(m->cbp&0x30){
        uint8_t *dest[2] = {dest_cb, dest_cr};

        idct_add = d->hdsp.h264_idct_add;
        idct_dc_add = d->hdsp.h264_idct_dc_add;
        for(i=16; i<16+8; i++){
            if(mrs->non_zero_count_cache[ scan8[i] ])
                idct_add   (dest[(i&4)>>2] + block_offset[i], m->mb + i*16, uvlinesize);
            else if(m->mb[i*16])
                idct_dc_add(dest[(i&4)>>2] + block_offset[i], m->mb + i*16, uvlinesize);
        }
    }

#if OMPSS
    backup_mb_border(m, dest_y, dest_cb, dest_cr, linesize, uvlinesize);
    if (mb_x+1 <d->mb_width){
        H264Mb *mr = m+1;
        memcpy(mr->left_border, m->left_border, sizeof(m->left_border));
    }
    if (mb_y +1 <d->mb_height){
        H264Mb *md = m + d->mb_width;
        memcpy(md->top_border, m->top_border, sizeof(m->top_border));
        if (mb_x>0){
            H264Mb *mdl = m + d->mb_width -1;
            memcpy(mdl->top_border_next, m->top_border_next, sizeof(m->top_border_next));
        }
    }
#else
    backup_mb_border(d, m, dest_y, dest_cb, dest_cr, linesize, uvlinesize);
    if (mb_y +1 <d->mb_height && d->top_next != d->top){
        memcpy(&d->top_next[mb_x],&d->top[mb_x], sizeof(TopBorder));
    }
#endif

    ff_h264_filter_mb(d, mrs, s, m, dest_y, dest_cb, dest_cr);
}

MBRecContext *get_mbrec_context(H264Context *h){
    MBRecContext *d = av_mallocz(sizeof(MBRecContext));

    ff_h264dsp_init(&d->hdsp);
    ff_h264_pred_init(&d->hpc);
    dsputil_init(&d->dsp);

#if !OMPSS
    d->mrs = av_mallocz(sizeof(MBRecState));
#endif
    d->hdsp.qpel_put= d->dsp.put_h264_qpel_pixels_tab;
    d->hdsp.qpel_avg= d->dsp.avg_h264_qpel_pixels_tab;
    d->mb_height = h->mb_height;
    d->mb_width  = h->mb_width;
    d->mb_stride  = h->mb_stride;
    d->b_stride  = h->b_stride;
    d->height = h->height;
    d->width  = h->width;
    d->linesize = h->width + EDGE_WIDTH*2;
    d->uvlinesize = d->linesize>>1;

    d->scratchpad_y = av_malloc(d->linesize*16*sizeof(uint8_t));
    d->scratchpad_cb= av_malloc(d->uvlinesize*8*sizeof(uint8_t));
    d->scratchpad_cr= av_malloc(d->uvlinesize*8*sizeof(uint8_t));

    for (int i=0; i<16; i++){
        d->block_offset[i]= 4*((scan8[i] - scan8[0])&7) + 4*d->linesize*((scan8[i] - scan8[0])>>3);
    }
    for (int i=0; i<4; i++){
        d->block_offset[16+i]=
        d->block_offset[20+i]= 4*((scan8[i] - scan8[0])&7) + 4*d->uvlinesize*((scan8[i] - scan8[0])>>3);
    }



    return d;
}

void free_mbrec_context(MBRecContext *d){
#if !OMPSS
    av_free(d->mrs);
#endif
    av_free(d->scratchpad_y);
    av_free(d->scratchpad_cb);
    av_free(d->scratchpad_cr);
    av_free(d);
}
