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

#include "h264_types.h"
#include "h264_data.h"

static inline void mc_dir_part(MBRecContext *d, MBRecState *mrs, H264Mb *m, DecodedPicture *pic, int n, int square,
                               int chroma_height, int delta, int list,uint8_t *dest_y,
                               uint8_t *dest_cb, uint8_t *dest_cr, int src_x_offset, int src_y_offset,
                               qpel_mc_func *qpix_op, h264_chroma_mc_func chroma_op){
    const int mx= mrs->mv_cache[list][ scan8[n] ][0] + src_x_offset*8;
    const int my= mrs->mv_cache[list][ scan8[n] ][1] + src_y_offset*8;
    const int luma_xy= (mx&3) + ((my&3)<<2);
    const int pic_width  = 16*d->mb_width;
    const int pic_height = 16*d->mb_height;

    uint8_t *src_y, *src_cb, *src_cr;
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

    src_y = pic->data[0] + ymx + ymy*d->linesize;
    qpix_op[luma_xy](dest_y, src_y, d->linesize); //FIXME try variable height perhaps?
    if(!square){
        qpix_op[luma_xy](dest_y + delta, src_y + delta, d->linesize);
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

    src_cb= pic->data[1] + cmx + cmy*d->uvlinesize;
    src_cr= pic->data[2] + cmx + cmy*d->uvlinesize;

    chroma_op(dest_cb, src_cb, d->uvlinesize, chroma_height, mx&7, my&7);
    chroma_op(dest_cr, src_cr, d->uvlinesize, chroma_height, mx&7, my&7);
                               }

static inline void mc_part_std(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m, int n, int square, int chroma_height, int delta,
                 uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,
                 int x_offset, int y_offset,
                 qpel_mc_func *qpix_put, h264_chroma_mc_func chroma_put,
                 qpel_mc_func *qpix_avg, h264_chroma_mc_func chroma_avg,
                 int list0, int list1){
    qpel_mc_func *qpix_op=  qpix_put;
    h264_chroma_mc_func chroma_op= chroma_put;

    dest_y  += 2*x_offset + 2*y_offset*d->  linesize;
    dest_cb +=   x_offset +   y_offset*d->uvlinesize;
    dest_cr +=   x_offset +   y_offset*d->uvlinesize;
    x_offset += 8*m->mb_x;
    y_offset += 8*m->mb_y;

    if(list0){
        DecodedPicture *ref= s->dp_ref_list[0][ mrs->ref_cache[0][ scan8[n] ] ];
        mc_dir_part(d, mrs, m, ref, n, square, chroma_height, delta, 0,
                    dest_y, dest_cb, dest_cr, x_offset, y_offset, qpix_op, chroma_op);

        qpix_op=  qpix_avg;
        chroma_op= chroma_avg;
    }

    if(list1){
        DecodedPicture *ref= s->dp_ref_list[1][ mrs->ref_cache[1][ scan8[n] ] ];
        mc_dir_part(d, mrs, m, ref, n, square, chroma_height, delta, 1,
                    dest_y, dest_cb, dest_cr, x_offset, y_offset, qpix_op, chroma_op);
    }
}

static inline void mc_part_weighted(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m, int n, int square, int chroma_height, int delta,
                      uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,
                      int x_offset, int y_offset,
                      qpel_mc_func *qpix_put, h264_chroma_mc_func chroma_put,
                      h264_weight_func luma_weight_op, h264_weight_func chroma_weight_op,
                      h264_biweight_func luma_weight_avg, h264_biweight_func chroma_weight_avg,
                      int list0, int list1){
    dest_y  += 2*x_offset + 2*y_offset*d->  linesize;
    dest_cb +=   x_offset +   y_offset*d->uvlinesize;
    dest_cr +=   x_offset +   y_offset*d->uvlinesize;
    x_offset += 8*m->mb_x;
    y_offset += 8*m->mb_y;

    if(list0 && list1){
        /* don't optimize for luma-only case, since B-frames usually
         * use implicit weights => chroma too. */
        uint8_t *tmp_y  = d->scratchpad_y  + 2*x_offset +16 ;
        uint8_t *tmp_cb = d->scratchpad_cb + x_offset + 8;
        uint8_t *tmp_cr = d->scratchpad_cr + x_offset + 8;

        /*
         *    uint8_t *tmp_cb = d->scratchpad;
         *    uint8_t *tmp_cr = d->scratchpad + 8;
         *    uint8_t *tmp_y  = d->scratchpad + 8*d->uvlinesize;*/
        int refn0 = mrs->ref_cache[0][ scan8[n] ];
        int refn1 = mrs->ref_cache[1][ scan8[n] ];

        mc_dir_part(d, mrs, m, s->dp_ref_list[0][refn0], n, square, chroma_height, delta, 0,
                    dest_y, dest_cb, dest_cr, x_offset, y_offset, qpix_put, chroma_put);
        mc_dir_part(d, mrs, m, s->dp_ref_list[1][refn1], n, square, chroma_height, delta, 1,
                    tmp_y, tmp_cb, tmp_cr, x_offset, y_offset, qpix_put, chroma_put);

        if(s->use_weight == 2){
            int weight0 = s->implicit_weight[refn0][refn1][m->mb_y&1];
            int weight1 = 64 - weight0;
            luma_weight_avg(  dest_y,  tmp_y,  d->  linesize, 5, weight0, weight1, 0);
            chroma_weight_avg(dest_cb, tmp_cb, d->uvlinesize, 5, weight0, weight1, 0);
            chroma_weight_avg(dest_cr, tmp_cr, d->uvlinesize, 5, weight0, weight1, 0);
        }else{
            luma_weight_avg(dest_y, tmp_y, d->linesize, s->luma_log2_weight_denom,
                            s->luma_weight[refn0][0][0] , s->luma_weight[refn1][1][0],
                            s->luma_weight[refn0][0][1] + s->luma_weight[refn1][1][1]);
            chroma_weight_avg(dest_cb, tmp_cb, d->uvlinesize, s->chroma_log2_weight_denom,
                              s->chroma_weight[refn0][0][0][0] , s->chroma_weight[refn1][1][0][0],
                              s->chroma_weight[refn0][0][0][1] + s->chroma_weight[refn1][1][0][1]);
            chroma_weight_avg(dest_cr, tmp_cr, d->uvlinesize, s->chroma_log2_weight_denom,
                              s->chroma_weight[refn0][0][1][0] , s->chroma_weight[refn1][1][1][0],
                              s->chroma_weight[refn0][0][1][1] + s->chroma_weight[refn1][1][1][1]);
        }
    }else{
        int list = list1 ? 1 : 0;
        int refn = mrs->ref_cache[list][ scan8[n] ];
        DecodedPicture *ref= s->dp_ref_list[list][refn];
        mc_dir_part(d, mrs, m, ref, n, square, chroma_height, delta, list,
                    dest_y, dest_cb, dest_cr, x_offset, y_offset, qpix_put, chroma_put);

        luma_weight_op(dest_y, d->linesize, s->luma_log2_weight_denom,
                       s->luma_weight[refn][list][0], s->luma_weight[refn][list][1]);
        if(s->use_weight_chroma){
            chroma_weight_op(dest_cb, d->uvlinesize, s->chroma_log2_weight_denom,
                             s->chroma_weight[refn][list][0][0], s->chroma_weight[refn][list][0][1]);
            chroma_weight_op(dest_cr, d->uvlinesize, s->chroma_log2_weight_denom,
                             s->chroma_weight[refn][list][1][0], s->chroma_weight[refn][list][1][1]);
        }
    }
}

static inline void mc_part(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m, int n, int square, int chroma_height, int delta,
             uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,
             int x_offset, int y_offset,
             qpel_mc_func *qpix_put, h264_chroma_mc_func chroma_put,
             qpel_mc_func *qpix_avg, h264_chroma_mc_func chroma_avg,
             h264_weight_func *weight_op, h264_biweight_func *weight_avg,
             int list0, int list1){
    if((s->use_weight==2 && list0 && list1
        && (s->implicit_weight[ mrs->ref_cache[0][scan8[n]] ][ mrs->ref_cache[1][scan8[n]] ][m->mb_y&1] != 32))
        || s->use_weight==1)
        mc_part_weighted(d, mrs, s, m, n, square, chroma_height, delta, dest_y, dest_cb, dest_cr,
                         x_offset, y_offset, qpix_put, chroma_put,
                         weight_op[0], weight_op[3], weight_avg[0], weight_avg[3], list0, list1);
    else
        mc_part_std(d, mrs, s, m, n, square, chroma_height, delta, dest_y, dest_cb, dest_cr,
                    x_offset, y_offset, qpix_put, chroma_put, qpix_avg, chroma_avg, list0, list1);
}

static inline void prefetch_motion(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m, int list){
    /* fetch pixels for estimated mv 4 macroblocks ahead
     * optimized for 64byte cache lines */
    const int refn = mrs->ref_cache[list][scan8[0]];

    if(refn >= 0){
        const int mx= (mrs->mv_cache[list][scan8[0]][0]>>2) + 16*m->mb_x + 8;
        const int my= (mrs->mv_cache[list][scan8[0]][1]>>2) + 16*m->mb_y;
        uint8_t **src= s->dp_ref_list[list][refn]->data;
        int off= mx + (my + (m->mb_x&3)*4)*d->linesize + 64;

        d->dsp.prefetch(src[0]+off, d->linesize, 4);
        off= (mx>>1) + ((my>>1) + (m->mb_x&7))*d->uvlinesize + 64;
        d->dsp.prefetch(src[1]+off, src[2]-src[1], 2);
    }
}

void hl_motion(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,
               qpel_mc_func (*qpix_put)[16], h264_chroma_mc_func (*chroma_put),
               qpel_mc_func (*qpix_avg)[16], h264_chroma_mc_func (*chroma_avg),
               h264_weight_func *weight_op, h264_biweight_func *weight_avg){
    const int mb_type= m->mb_type;
    assert(IS_INTER(mb_type));

    if (mb_type & MB_TYPE_L0)
        prefetch_motion(d, mrs, s, m, 0);
    if (mb_type & MB_TYPE_L1)
        prefetch_motion(d, mrs, s, m, 1);

    if(IS_16X16(mb_type)){
        mc_part(d, mrs, s, m, 0, 1, 8, 0, dest_y, dest_cb, dest_cr, 0, 0,
                qpix_put[0], chroma_put[0], qpix_avg[0], chroma_avg[0],
                weight_op, weight_avg,
                IS_DIR(mb_type, 0, 0), IS_DIR(mb_type, 0, 1));
    }else if(IS_16X8(mb_type)){
        mc_part(d, mrs, s, m, 0, 0, 4, 8, dest_y, dest_cb, dest_cr, 0, 0,
                qpix_put[1], chroma_put[0], qpix_avg[1], chroma_avg[0],
                &weight_op[1], &weight_avg[1],
                IS_DIR(mb_type, 0, 0), IS_DIR(mb_type, 0, 1));
        mc_part(d, mrs, s, m, 8, 0, 4, 8, dest_y, dest_cb, dest_cr, 0, 4,
                qpix_put[1], chroma_put[0], qpix_avg[1], chroma_avg[0],
                &weight_op[1], &weight_avg[1],
                IS_DIR(mb_type, 1, 0), IS_DIR(mb_type, 1, 1));
    }else if(IS_8X16(mb_type)){
        mc_part(d, mrs, s, m, 0, 0, 8, 8*d->linesize, dest_y, dest_cb, dest_cr, 0, 0,
                qpix_put[1], chroma_put[1], qpix_avg[1], chroma_avg[1],
                &weight_op[2], &weight_avg[2],
                IS_DIR(mb_type, 0, 0), IS_DIR(mb_type, 0, 1));
        mc_part(d, mrs, s, m, 4, 0, 8, 8*d->linesize, dest_y, dest_cb, dest_cr, 4, 0,
                qpix_put[1], chroma_put[1], qpix_avg[1], chroma_avg[1],
                &weight_op[2], &weight_avg[2],
                IS_DIR(mb_type, 1, 0), IS_DIR(mb_type, 1, 1));
    }else{
        int i;

        assert(IS_8X8(mb_type));

        for(i=0; i<4; i++){
            const int sub_mb_type= m->sub_mb_type[i];
            const int n= 4*i;
            int x_offset= (i&1)<<2;
            int y_offset= (i&2)<<1;

            if(IS_SUB_8X8(sub_mb_type)){
                mc_part(d, mrs, s, m, n, 1, 4, 0, dest_y, dest_cb, dest_cr, x_offset, y_offset,
                        qpix_put[1], chroma_put[1], qpix_avg[1], chroma_avg[1],
                        &weight_op[3], &weight_avg[3],
                        IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
            }else if(IS_SUB_8X4(sub_mb_type)){
                mc_part(d, mrs, s, m, n, 0, 2, 4, dest_y, dest_cb, dest_cr, x_offset, y_offset,
                        qpix_put[2], chroma_put[1], qpix_avg[2], chroma_avg[1],
                        &weight_op[4], &weight_avg[4],
                        IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
                mc_part(d, mrs, s, m, n+2, 0, 2, 4, dest_y, dest_cb, dest_cr, x_offset, y_offset+2,
                        qpix_put[2], chroma_put[1], qpix_avg[2], chroma_avg[1],
                        &weight_op[4], &weight_avg[4],
                        IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
            }else if(IS_SUB_4X8(sub_mb_type)){
                mc_part(d, mrs, s, m, n, 0, 4, 4*d->linesize, dest_y, dest_cb, dest_cr, x_offset, y_offset,
                        qpix_put[2], chroma_put[2], qpix_avg[2], chroma_avg[2],
                        &weight_op[5], &weight_avg[5],
                        IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
                mc_part(d, mrs, s, m, n+1, 0, 4, 4*d->linesize, dest_y, dest_cb, dest_cr, x_offset+2, y_offset,
                        qpix_put[2], chroma_put[2], qpix_avg[2], chroma_avg[2],
                        &weight_op[5], &weight_avg[5],
                        IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
            }else{
                int j;
                assert(IS_SUB_4X4(sub_mb_type));
                for(j=0; j<4; j++){
                    int sub_x_offset= x_offset + 2*(j&1);
                    int sub_y_offset= y_offset +   (j&2);
                    mc_part(d, mrs, s, m, n+j, 1, 2, 0, dest_y, dest_cb, dest_cr, sub_x_offset, sub_y_offset,
                            qpix_put[2], chroma_put[2], qpix_avg[2], chroma_avg[2],
                            &weight_op[6], &weight_avg[6],
                            IS_DIR(sub_mb_type, 0, 0), IS_DIR(sub_mb_type, 0, 1));
                }
            }
        }
    }
}
