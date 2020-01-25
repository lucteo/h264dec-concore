/*
 * H.26L/H.264/AVC/JVT/14496-10/... loop filter
 * Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * H.264 / AVC / MPEG4 part10 loop filter.
 * @author Michael Niedermayer <michaelni@gmx.at>
 * @author Chi Ching Chi <chi.c.chi@tu-berlin.de>
 */

#include "dsputil.h"
#include "mathops.h"
#include "rectangle.h"
#include "h264_types.h"
#include "h264_misc.h"
#include "h264_data.h"
//#undef NDEBUG
#include <assert.h>

/* Deblocking filter (p153) */
static const uint8_t alpha_table[52*3] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  4,  4,  5,  6,
     7,  8,  9, 10, 12, 13, 15, 17, 20, 22,
    25, 28, 32, 36, 40, 45, 50, 56, 63, 71,
    80, 90,101,113,127,144,162,182,203,226,
   255,255,
   255,255,255,255,255,255,255,255,255,255,255,255,255,
   255,255,255,255,255,255,255,255,255,255,255,255,255,
   255,255,255,255,255,255,255,255,255,255,255,255,255,
   255,255,255,255,255,255,255,255,255,255,255,255,255,
};
static const uint8_t beta_table[52*3] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  2,  2,  2,  3,
     3,  3,  3,  4,  4,  4,  6,  6,  7,  7,
     8,  8,  9,  9, 10, 10, 11, 11, 12, 12,
    13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
    18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
};
static const uint8_t tc0_table[52*3][4] = {
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 },
    {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 0 }, {-1, 0, 0, 1 },
    {-1, 0, 0, 1 }, {-1, 0, 0, 1 }, {-1, 0, 0, 1 }, {-1, 0, 1, 1 }, {-1, 0, 1, 1 }, {-1, 1, 1, 1 },
    {-1, 1, 1, 1 }, {-1, 1, 1, 1 }, {-1, 1, 1, 1 }, {-1, 1, 1, 2 }, {-1, 1, 1, 2 }, {-1, 1, 1, 2 },
    {-1, 1, 1, 2 }, {-1, 1, 2, 3 }, {-1, 1, 2, 3 }, {-1, 2, 2, 3 }, {-1, 2, 2, 4 }, {-1, 2, 3, 4 },
    {-1, 2, 3, 4 }, {-1, 3, 3, 5 }, {-1, 3, 4, 6 }, {-1, 3, 4, 6 }, {-1, 4, 5, 7 }, {-1, 4, 5, 8 },
    {-1, 4, 6, 9 }, {-1, 5, 7,10 }, {-1, 6, 8,11 }, {-1, 6, 8,13 }, {-1, 7,10,14 }, {-1, 8,11,16 },
    {-1, 9,12,18 }, {-1,10,13,20 }, {-1,11,15,23 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
    {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 }, {-1,13,17,25 },
};

av_always_inline static void filter_mb_edgev( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, MBRecContext *mrc, H264Slice *s) {
    const unsigned int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
        tc[0] = tc0_table[index_a][bS[0]];
        tc[1] = tc0_table[index_a][bS[1]];
        tc[2] = tc0_table[index_a][bS[2]];
        tc[3] = tc0_table[index_a][bS[3]];
        mrc->hdsp.h264_h_loop_filter_luma(pix, stride, alpha, beta, tc);
    } else {
        mrc->hdsp.h264_h_loop_filter_luma_intra(pix, stride, alpha, beta);
    }
}

av_always_inline static void filter_mb_edgecv( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, MBRecContext *mrc, H264Slice *s ) {
    const unsigned int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
        tc[0] = tc0_table[index_a][bS[0]]+1;
        tc[1] = tc0_table[index_a][bS[1]]+1;
        tc[2] = tc0_table[index_a][bS[2]]+1;
        tc[3] = tc0_table[index_a][bS[3]]+1;
        mrc->hdsp.h264_h_loop_filter_chroma(pix, stride, alpha, beta, tc);
    } else {
        mrc->hdsp.h264_h_loop_filter_chroma_intra(pix, stride, alpha, beta);
    }
}


av_always_inline static void filter_mb_edgeh( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, MBRecContext *mrc, H264Slice *s ) {
    const unsigned int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
        tc[0] = tc0_table[index_a][bS[0]];
        tc[1] = tc0_table[index_a][bS[1]];
        tc[2] = tc0_table[index_a][bS[2]];
        tc[3] = tc0_table[index_a][bS[3]];
        mrc->hdsp.h264_v_loop_filter_luma(pix, stride, alpha, beta, tc);
    } else {
        mrc->hdsp.h264_v_loop_filter_luma_intra(pix, stride, alpha, beta);
    }
}

av_always_inline static void filter_mb_edgech( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, MBRecContext *mrc, H264Slice *s ) {
    const unsigned int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
        tc[0] = tc0_table[index_a][bS[0]]+1;
        tc[1] = tc0_table[index_a][bS[1]]+1;
        tc[2] = tc0_table[index_a][bS[2]]+1;
        tc[3] = tc0_table[index_a][bS[3]]+1;
        mrc->hdsp.h264_v_loop_filter_chroma(pix, stride, alpha, beta, tc);
    } else {
        mrc->hdsp.h264_v_loop_filter_chroma_intra(pix, stride, alpha, beta);
    }
}

static av_always_inline void filter_mb_dir(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, int dir) {
    const int mbm_type = dir == 0 ? mrs->left_type : mrs->top_type;
    const int qp_xy= m->qscale_mb_xy;
    const int qp_dir = dir == 0 ? m->qscale_left_mb_xy : m->qscale_top_mb_xy;
    const int linesize = mrc->linesize;
    const int uvlinesize = mrc->uvlinesize;
    const int mb_type = m->mb_type;
    int edge;
    const int edges = mrs->edges[dir];

    if(mbm_type){
        int16_t* bS=mrs->bS[dir][0];
        /* Filter edge */
        // Do not use s->qscale as luma quantizer because it has not the same
        // value in IPCM macroblocks.
        if(bS[0]+bS[1]+bS[2]+bS[3]){
            int qp = ( qp_xy + qp_dir + 1 ) >> 1;
            if( dir == 0 ) {
                filter_mb_edgev( &img_y[0], linesize, bS, qp, mrc, s );
                {
                    int qp= ( get_chroma_qp(s, 0, qp_xy) + get_chroma_qp( s, 0, qp_dir) + 1 ) >> 1;
                    filter_mb_edgecv( &img_cb[0], uvlinesize, bS, qp, mrc, s);
                    filter_mb_edgecv( &img_cr[0], uvlinesize, bS, qp, mrc, s);
                }
            } else {
                filter_mb_edgeh( &img_y[0], linesize, bS, qp, mrc, s );
                {
                    int qp= ( get_chroma_qp(s, 0, qp_xy) + get_chroma_qp( s, 0, qp_dir) + 1 ) >> 1;
                    filter_mb_edgech( &img_cb[0], uvlinesize, bS, qp, mrc, s);
                    filter_mb_edgech( &img_cr[0], uvlinesize, bS, qp, mrc, s);
                }
            }
        }
    }

    for( edge = 1; edge < edges; edge++ ) {
        int16_t* bS=mrs->bS[dir][edge];
        int qp = qp_xy;

        if( IS_8x8DCT(mb_type & (edge<<24)) ) // (edge&1) && IS_8x8DCT(mb_type)
            continue;

        if(bS[0]+bS[1]+bS[2]+bS[3] == 0)
            continue;

        /* Filter edge */
        // Do not use s->qscale as luma quantizer because it has not the same
        // value in IPCM macroblocks.

        if( dir == 0 ) {
            filter_mb_edgev( &img_y[4*edge], linesize, bS, qp, mrc, s);
            if( (edge&1) == 0 ) {
                filter_mb_edgecv( &img_cb[2*edge], uvlinesize, bS, get_chroma_qp(s, 0, qp_xy), mrc, s);
                filter_mb_edgecv( &img_cr[2*edge], uvlinesize, bS, get_chroma_qp(s, 1, qp_xy), mrc, s);
            }
        } else {
            filter_mb_edgeh( &img_y[4*edge*linesize], linesize, bS, qp, mrc, s );
            if( (edge&1) == 0 ) {
                filter_mb_edgech( &img_cb[2*edge*uvlinesize], uvlinesize, bS, get_chroma_qp(s, 0, qp_xy), mrc, s);
                filter_mb_edgech( &img_cr[2*edge*uvlinesize], uvlinesize, bS, get_chroma_qp(s, 1, qp_xy), mrc, s);
            }
        }
    }
}

static int check_mv(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, long b_idx, long bn_idx, int mvy_limit){
    int v;
    v= mrs->ref_cache[0][b_idx] != mrs->ref_cache[0][bn_idx];
    if(!v && mrs->ref_cache[0][b_idx]!=-1)
        // absolute value >= 7 | ...
        v= ((unsigned) (mrs->mv_cache[0][b_idx][0] - mrs->mv_cache[0][bn_idx][0] + 3) >= 7U) |
        ((FFABS( mrs->mv_cache[0][b_idx][1] - mrs->mv_cache[0][bn_idx][1] )) >= mvy_limit);

    if(s->list_count==2){
        if(!v)
            v = (mrs->ref_cache[1][b_idx] != mrs->ref_cache[1][bn_idx]) |
            ((unsigned) (mrs->mv_cache[1][b_idx][0] - mrs->mv_cache[1][bn_idx][0] + 3) >= 7U) |
            ((FFABS( mrs->mv_cache[1][b_idx][1] - mrs->mv_cache[1][bn_idx][1] )) >= mvy_limit);

        if(v){
            if((mrs->ref_cache[0][b_idx] != mrs->ref_cache[1][bn_idx]) |
                (mrs->ref_cache[1][b_idx] != mrs->ref_cache[0][bn_idx]))
                return 1;
            return
            ((unsigned) (mrs->mv_cache[0][b_idx][0] - mrs->mv_cache[1][bn_idx][0] + 3) >= 7U) |
            ((FFABS( mrs->mv_cache[0][b_idx][1] - mrs->mv_cache[1][bn_idx][1] )) >= mvy_limit) |
            ((unsigned) (mrs->mv_cache[1][b_idx][0] - mrs->mv_cache[0][bn_idx][0] + 3) >= 7U) |
            ((FFABS( mrs->mv_cache[1][b_idx][1] - mrs->mv_cache[0][bn_idx][1] )) >= mvy_limit);
        }
    }

    return v;
}

static void calc_bS_values(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int mvy_limit, int dir) {
    int mb_type = m->mb_type;
    int edge;
    const int mbm_type = dir == 0 ? mrs->left_type : mrs->top_type;

    // how often to recheck mv-based bS when iterating between edges
    static const uint8_t mask_edge_tab[2][8]={{0,3,3,3,1,1,1,1},
    {0,3,1,1,3,3,3,3}};
    const int mask_edge = mask_edge_tab[dir][(mb_type>>3)&7];
    const int edges = mask_edge== 3 && !(m->cbp&15) ? 1 : 4;
    // how often to recheck mv-based bS when iterating along each edge
    const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));

    mrs->edges[dir]= edges;

    if(mbm_type){
        int16_t* bS=mrs->bS[dir][0];
        if( IS_INTRA(mb_type|mbm_type)) {
            AV_WN64A(bS, 0x0004000400040004ULL);
        } else {
            int i;
            int mv_done;
            if( mask_par0 && ((mbm_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {
                int b_idx= 8 + 4;
                int bn_idx= b_idx - (dir ? 8:1);

                bS[0] = bS[1] = bS[2] = bS[3] = check_mv(mrc, mrs, s, 8 + 4, bn_idx, mvy_limit);
                mv_done = 1;
            }
            else
                mv_done = 0;

            for( i = 0; i < 4; i++ ) {
                int x = dir == 0 ? 0 : i;
                int y = dir == 0 ? i    : 0;
                int b_idx= 8 + 4 + x + 8*y;
                int bn_idx= b_idx - (dir ? 8:1);

                if( mrs->non_zero_count_cache[b_idx] |
                    mrs->non_zero_count_cache[bn_idx] ) {
                    bS[i] = 2;
                }
                else if(!mv_done)
                {
                    bS[i] = check_mv(mrc, mrs, s, b_idx, bn_idx, mvy_limit);
                }
            }
        }
    }

    /* Calculate bS */
    for( edge = 1; edge < edges; edge++ ) {
        int16_t* bS=mrs->bS[dir][edge];

        if( IS_8x8DCT(mb_type & (edge<<24)) ) // (edge&1) && IS_8x8DCT(mb_type)
            continue;

        if( IS_INTRA(mb_type)) {
            AV_WN64A(bS, 0x0003000300030003ULL);
        } else {
            int i;
            int mv_done;

            if( edge & mask_edge ) {
                AV_ZERO64(bS);
                mv_done = 1;
            }
            else if( mask_par0 ) {
                int b_idx= 8 + 4 + edge * (dir ? 8:1);
                int bn_idx= b_idx - (dir ? 8:1);

                bS[0] = bS[1] = bS[2] = bS[3] = check_mv(mrc, mrs, s, b_idx, bn_idx, mvy_limit);
                mv_done = 1;
            }
            else
                mv_done = 0;

            for( i = 0; i < 4; i++ ) {
                int x = dir == 0 ? edge : i;
                int y = dir == 0 ? i    : edge;
                int b_idx= 8 + 4 + x + 8*y;
                int bn_idx= b_idx - (dir ? 8:1);

                if( mrs->non_zero_count_cache[b_idx] |
                    mrs->non_zero_count_cache[bn_idx] ) {
                    bS[i] = 2;
                }
                else if(!mv_done)
                {
                    bS[i] = check_mv(mrc, mrs, s, b_idx, bn_idx, mvy_limit);
                }
            }

            if(bS[0]+bS[1]+bS[2]+bS[3] == 0)
                continue;
        }

    }
}


/**
*
* @return zero if the loop filter can be skiped
*/
static int fill_filter_caches(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int mb_type){
    H264Mb *m_top = m - mrc->mb_width;
    H264Mb *m_left = m - 1;
    const int mb_x = m->mb_x;
    const int mb_y = m->mb_y;
    int top_type, left_type;
    int qp, top_qp, left_qp;
    int qp_thresh = s->qp_thresh; //FIXME strictly we should store qp_thresh for each mb of a slice

    qp = m->qscale_mb_xy ;
    left_qp = m->qscale_left_mb_xy ;
    top_qp  = m->qscale_top_mb_xy ;

    //for sufficiently low qp, filtering wouldn't do anything
    //this is a conservative estimate: could also check beta_offset and more accurate chroma_qp
    if(qp <= qp_thresh
        && (!(mb_x+mb_y) || ((qp + left_qp + 1)>>1) <= qp_thresh)
        && ( mb_y==0 || ((qp + top_qp + 1)>>1) <= qp_thresh)){
        return 0;
    }

    if(IS_INTRA(mb_type)){
        return 1;
    }

    {
        int list;
        for(list=0; list<s->list_count; list++){
            int8_t *ref;

            if(!USES_LIST(mb_type, list)){
                fill_rectangle( mrs->mv_cache[list][scan8[0]], 4, 4, 8, pack16to32(0,0), 4);
                fill_rectangle( mrs->mv_cache[list][scan8[0]], 4, 4, 8, pack16to32(0,0), 4);
                AV_WN32A(&mrs->ref_cache[list][scan8[ 0]], ((LIST_NOT_USED)&0xFF)*0x01010101u);
                AV_WN32A(&mrs->ref_cache[list][scan8[ 2]], ((LIST_NOT_USED)&0xFF)*0x01010101u);
                AV_WN32A(&mrs->ref_cache[list][scan8[ 8]], ((LIST_NOT_USED)&0xFF)*0x01010101u);
                AV_WN32A(&mrs->ref_cache[list][scan8[10]], ((LIST_NOT_USED)&0xFF)*0x01010101u);
                continue;
            }

            ref = &mrs->ref_index[list][4*mb_x];
            {
                int (*ref2frm)[64] =(void *) (s->ref2frm[0] +  2);
                AV_WN32A(&mrs->ref_cache[list][scan8[ 0]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);
                AV_WN32A(&mrs->ref_cache[list][scan8[ 2]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);
                ref += 2;

                AV_WN32A(&mrs->ref_cache[list][scan8[ 8]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);
                AV_WN32A(&mrs->ref_cache[list][scan8[10]], (pack16to32(ref2frm[list][ref[0]],ref2frm[list][ref[1]])&0x00FF00FF)*0x0101);
            }
        }
    }

    /*
    0 . T T. T T T T
    1 L . .L . . . .
    2 L . .L . . . .
    3 . T TL . . . .
    4 L . .L . . . .
    5 L . .. . . . .
    */

    if (IS_SKIP(mb_type)){
        memset(mrs->non_zero_count_cache + 8, 0, 8*5); //FIXME ugly, remove pfui
    }

    //FIXME constraint_intra_pred & partitioning & nnz (let us hope this is just a typo in the spec)
    top_type  = mrs->top_type;
    left_type = mrs->left_type;
    if(top_type){
        AV_COPY32(&mrs->non_zero_count_cache[4+8*0], &m_top->non_zero_count[3*4]);
    }

    if(left_type){
        mrs->non_zero_count_cache[3+8*1]= m_left->non_zero_count[3+0*4];
        mrs->non_zero_count_cache[3+8*2]= m_left->non_zero_count[3+1*4];
        mrs->non_zero_count_cache[3+8*3]= m_left->non_zero_count[3+2*4];
        mrs->non_zero_count_cache[3+8*4]= m_left->non_zero_count[3+3*4];
    }

    if(IS_INTER(mb_type) || IS_DIRECT(mb_type)){
        int list;
        for(list=0; list<s->list_count; list++){
            if(USES_LIST(top_type, list)){
                const int b_xy= 4*mb_x + 3*mrc->b_stride;
                const int b8_x= 4*mb_x + 2;
                int (*ref2frm)[64] = (void *) (s->ref2frm[0] +  2);
                AV_COPY128(mrs->mv_cache[list][scan8[0] + 0 - 1*8], mrs->motion_val_top[list][b_xy + 0]);

                mrs->ref_cache[list][scan8[0] + 0 - 1*8]=
                mrs->ref_cache[list][scan8[0] + 1 - 1*8]= ref2frm[list][mrs->ref_index_top[list][b8_x + 0]];
                mrs->ref_cache[list][scan8[0] + 2 - 1*8]=
                mrs->ref_cache[list][scan8[0] + 3 - 1*8]= ref2frm[list][mrs->ref_index_top[list][b8_x + 1]];
            }else{
                AV_ZERO128(mrs->mv_cache[list][scan8[0] + 0 - 1*8]);
                AV_WN32A(&mrs->ref_cache[list][scan8[0] + 0 - 1*8], ((LIST_NOT_USED)&0xFF)*0x01010101u);
            }

            if(USES_LIST(left_type, list)){
                const int b_x = 4*(mb_x-1) + 3;
                const int b8_x= 4*(mb_x-1) + 1;
                int (*ref2frm)[64] = (void *) (s->ref2frm[0] +  2);
                AV_COPY32(mrs->mv_cache[list][scan8[0] - 1 + 0 ], mrs->motion_val[list][b_x + mrc->b_stride*0]);
                AV_COPY32(mrs->mv_cache[list][scan8[0] - 1 + 8 ], mrs->motion_val[list][b_x + mrc->b_stride*1]);
                AV_COPY32(mrs->mv_cache[list][scan8[0] - 1 +16 ], mrs->motion_val[list][b_x + mrc->b_stride*2]);
                AV_COPY32(mrs->mv_cache[list][scan8[0] - 1 +24 ], mrs->motion_val[list][b_x + mrc->b_stride*3]);

                mrs->ref_cache[list][scan8[0] - 1 + 0 ]=
                mrs->ref_cache[list][scan8[0] - 1 + 8 ]= ref2frm[list][mrs->ref_index[list][b8_x + 2*0]];
                mrs->ref_cache[list][scan8[0] - 1 +16 ]=
                mrs->ref_cache[list][scan8[0] - 1 +24 ]= ref2frm[list][mrs->ref_index[list][b8_x + 2*1]];

            }else{
                AV_ZERO32(mrs->mv_cache [list][scan8[0] - 1 + 0 ]);
                AV_ZERO32(mrs->mv_cache [list][scan8[0] - 1 + 8 ]);
                AV_ZERO32(mrs->mv_cache [list][scan8[0] - 1 +16 ]);
                AV_ZERO32(mrs->mv_cache [list][scan8[0] - 1 +24 ]);

                mrs->ref_cache[list][scan8[0] - 1 + 0  ]=
                mrs->ref_cache[list][scan8[0] - 1 + 8  ]=
                mrs->ref_cache[list][scan8[0] - 1 + 16 ]=
                mrs->ref_cache[list][scan8[0] - 1 + 24 ]= LIST_NOT_USED;
            }
        }
    }
    return 1;
}

void ff_h264_filter_mb(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr) {
    if (fill_filter_caches(mrc, mrs, s, m, m->mb_type)){
        calc_bS_values(mrc, mrs, s, m, 4, 0);
        calc_bS_values(mrc, mrs, s, m, 4, 1);
        filter_mb_dir(mrc, mrs, s, m, img_y, img_cb, img_cr, 0);
        filter_mb_dir(mrc, mrs, s, m, img_y, img_cb, img_cr, 1);
    }
}
