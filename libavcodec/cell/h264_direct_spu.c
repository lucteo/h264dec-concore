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
#include "libavcodec/avcodec.h"
#include "dsputil_spu.h"
#include "h264_tables.h"
#include "h264_types_spu.h"
#include "libavutil/common.h"
#include "libavutil/intreadwrite.h"
#include "mathops_spu.h"
#include "rectangle_spu.h"

//#undef NDEBUG
#include <assert.h>
static void pred_spatial_direct_motion(H264Cabac_spu *hc, EDSlice_spu *s, int *mb_type){
    H264Mb *m = s->m;
    int b4_stride = hc->b_stride;
	const int mb_x = m->mb_x;
    int mb_type_col[2];
    const int16_t (*l1mv0)[2], (*l1mv1)[2];
    const int8_t *l1ref0, *l1ref1;
    const int is_b8x8 = IS_8X8(*mb_type);
    unsigned int sub_mb_type= MB_TYPE_L0L1;
    int i8, i4;
    int ref[2];
    int mv[2];
    int list;

    //assert(h->ref_list[1][0].reference&3);

#define MB_TYPE_16x16_OR_INTRA (MB_TYPE_16x16|MB_TYPE_INTRA4x4|MB_TYPE_INTRA16x16|MB_TYPE_INTRA_PCM)

    /* ref = min(neighbors) */
    for(list=0; list<2; list++){
        int left_ref = m->ref_cache[list][scan8[0] - 1];
        int top_ref  = m->ref_cache[list][scan8[0] - 8];
        int refc = m->ref_cache[list][scan8[0] - 8 + 4];
        const int16_t *C= m->mv_cache[list][ scan8[0] - 8 + 4];
        if(refc == PART_NOT_AVAILABLE){
            refc = m->ref_cache[list][scan8[0] - 8 - 1];
            C    = m-> mv_cache[list][scan8[0] - 8 - 1];
        }
        ref[list] = FFMIN3((unsigned)left_ref, (unsigned)top_ref, (unsigned)refc);
        if(ref[list] >= 0){
            //this is just pred_motion() but with the cases removed that cannot happen for direct blocks
            const int16_t * const A= m->mv_cache[list][ scan8[0] - 1 ];
            const int16_t * const B= m->mv_cache[list][ scan8[0] - 8 ];

            int match_count= (left_ref==ref[list]) + (top_ref==ref[list]) + (refc==ref[list]);
            if(match_count > 1){ //most common
                mv[list]= pack16to32(mid_pred(A[0], B[0], C[0]),
                                     mid_pred(A[1], B[1], C[1]) );
            }else {
                assert(match_count==1);
                if(left_ref==ref[list]){
                    mv[list]= AV_RN32A(A);
                }else if(top_ref==ref[list]){
                    mv[list]= AV_RN32A(B);
                }else{
                    mv[list]= AV_RN32A(C);
                }
            }
        }else{
            int mask= ~(MB_TYPE_L0 << (2*list));
            mv[list] = 0;
            ref[list] = -1;
            if(!is_b8x8)
                *mb_type &= mask;
            sub_mb_type &= mask;
        }
    }

    if(ref[0] < 0 && ref[1] < 0){
        ref[0] = ref[1] = 0;
        if(!is_b8x8)
            *mb_type |= MB_TYPE_L0L1;
        sub_mb_type |= MB_TYPE_L0L1;
    }

    if(!(is_b8x8|mv[0]|mv[1])){
        fill_rectangle(&m->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);
        fill_rectangle(&m->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);
        fill_rectangle(&m->mv_cache[0][scan8[0]], 4, 4, 8, 0, 4);
        fill_rectangle(&m->mv_cache[1][scan8[0]], 4, 4, 8, 0, 4);
        *mb_type= (*mb_type & ~(MB_TYPE_8x8|MB_TYPE_16x8|MB_TYPE_8x16|MB_TYPE_P1L0|MB_TYPE_P1L1))|MB_TYPE_16x16|MB_TYPE_DIRECT2;
        return;
    }

    mb_type_col[0] =
    mb_type_col[1] = hc->list1_mb_type[mb_x];

    sub_mb_type |= MB_TYPE_16x16|MB_TYPE_DIRECT2; /* B_SUB_8x8 */
    if(!is_b8x8 && (mb_type_col[0] & MB_TYPE_16x16_OR_INTRA)){
        *mb_type   |= MB_TYPE_16x16|MB_TYPE_DIRECT2; /* B_16x16 */
    }else if(!is_b8x8 && (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16))){
        *mb_type   |= MB_TYPE_DIRECT2 | (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16));
    }else{
        if(!s->direct_8x8_inference_flag){
            /* FIXME save sub mb types from previous frames (or derive from MVs)
            * so we know exactly what block size to use */
            sub_mb_type += (MB_TYPE_8x8-MB_TYPE_16x16); /* B_SUB_4x4 */
        }
        *mb_type   |= MB_TYPE_8x8;
    }

//     l1mv0  = (void *) &hc->list1_motion_val[0][4*mb_x];
//     l1mv1  = (void *) &hc->list1_motion_val[1][4*mb_x];
	l1mv0  = (void *) hc->list1_motion_val[0];
    l1mv1  = (void *) hc->list1_motion_val[1];
    l1ref0 = &hc->list1_ref_index [0][4*mb_x];
    l1ref1 = &hc->list1_ref_index [1][4*mb_x];
//     if(!b8_stride){
//         if(m->mb_y&1){
//             l1ref0 += 2;
//             l1ref1 += 2;
//             l1mv0  +=  2*b4_stride;
//             l1mv1  +=  2*b4_stride;
//         }
//     }

    if(IS_16X16(*mb_type)){
        int a,b;

        fill_rectangle(&m->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);
        fill_rectangle(&m->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);
        if(!IS_INTRA(mb_type_col[0]) && (   (l1ref0[0] == 0 && FFABS(l1mv0[0][0]) <= 1 && FFABS(l1mv0[0][1]) <= 1)
            || (l1ref0[0] < 0 && l1ref1[0] == 0 && FFABS(l1mv1[0][0]) <= 1 && FFABS(l1mv1[0][1]) <= 1
            ))){
            a=b=0;
            if(ref[0] > 0)
                a= mv[0];
            if(ref[1] > 0)
                b= mv[1];
        }else{
            a= mv[0];
            b= mv[1];
        }
        fill_rectangle(&m->mv_cache[0][scan8[0]], 4, 4, 8, a, 4);
        fill_rectangle(&m->mv_cache[1][scan8[0]], 4, 4, 8, b, 4);
    }else{
        int n=0;
        for(i8=0; i8<4; i8++){
            const int x8 = i8&1;
            const int y8 = i8>>1;

            if(is_b8x8 && !IS_DIRECT(m->sub_mb_type[i8]))
                continue;
            m->sub_mb_type[i8] = sub_mb_type;

            fill_rectangle(&m->mv_cache[0][scan8[i8*4]], 2, 2, 8, mv[0], 4);
            fill_rectangle(&m->mv_cache[1][scan8[i8*4]], 2, 2, 8, mv[1], 4);
            fill_rectangle(&m->ref_cache[0][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[0], 1);
            fill_rectangle(&m->ref_cache[1][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[1], 1);

            /* col_zero_flag */
            if(!IS_INTRA(mb_type_col[0]) && (l1ref0[i8] == 0 || (l1ref0[i8] < 0 && l1ref1[i8] == 0 ))
                ){
                const int16_t (*l1mv)[2]= l1ref0[i8] == 0 ? l1mv0 : l1mv1;
                if(IS_SUB_8X8(sub_mb_type)){
//                     const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];
					const int16_t *mv_col = l1mv[x8*3 + y8*3*4];
                    if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){
                        if(ref[0] == 0)
                            fill_rectangle(&m->mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);
                        if(ref[1] == 0)
                            fill_rectangle(&m->mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);
                        n+=4;
                    }
                }else{
                    int k=0;
                    for(i4=0; i4<4; i4++){
                        //const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];
						const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*4];
                        if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){
                            if(ref[0] == 0)
                                AV_ZERO32(m->mv_cache[0][scan8[i8*4+i4]]);
                            if(ref[1] == 0)
                                AV_ZERO32(m->mv_cache[1][scan8[i8*4+i4]]);
                            k++;
                        }
                    }
                    if(!(k&3))
                        m->sub_mb_type[i8]+= MB_TYPE_16x16 - MB_TYPE_8x8;
                    n+=k;
                }
            }
        }
        if(!is_b8x8 && !(n&15)){
            *mb_type= (*mb_type & ~(MB_TYPE_8x8|MB_TYPE_16x8|MB_TYPE_8x16|MB_TYPE_P1L0|MB_TYPE_P1L1))|MB_TYPE_16x16|MB_TYPE_DIRECT2;
        }
    }
}

static void pred_temp_direct_motion(H264Cabac_spu *hc, EDSlice_spu *s, int *mb_type){
    H264Mb *m = s->m;
	const int mb_x = m->mb_x;
    int b4_stride = hc->b_stride;
    int mb_type_col[2];
    const int16_t (*l1mv0)[2], (*l1mv1)[2];
    const int8_t *l1ref0, *l1ref1;
    const int is_b8x8 = IS_8X8(*mb_type);
    unsigned int sub_mb_type;
    int i8, i4;
    const int *map_col_to_list0[2] = {s->map_col_to_list0[0], s->map_col_to_list0[1]};
    const int *dist_scale_factor = s->dist_scale_factor;

    mb_type_col[0] =
    mb_type_col[1] = hc->list1_mb_type[mb_x];

    sub_mb_type = MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_8x8 */
    if(!is_b8x8 && (mb_type_col[0] & MB_TYPE_16x16_OR_INTRA)){
        *mb_type   |= MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_16x16 */
    }else if(!is_b8x8 && (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16))){
        *mb_type   |= MB_TYPE_L0L1|MB_TYPE_DIRECT2 | (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16));
    }else{
        if(!s->direct_8x8_inference_flag){
            /* FIXME save sub mb types from previous frames (or derive from MVs)
            * so we know exactly what block size to use */
            sub_mb_type = MB_TYPE_8x8|MB_TYPE_P0L0|MB_TYPE_P0L1|MB_TYPE_DIRECT2; /* B_SUB_4x4 */
        }
        *mb_type   |= MB_TYPE_8x8|MB_TYPE_L0L1;
    }

//     l1mv0  = (void *) &hc->list1_motion_val[0][4*mb_x];
//     l1mv1  = (void *) &hc->list1_motion_val[1][4*mb_x];
	l1mv0  = (void *) hc->list1_motion_val[0];
    l1mv1  = (void *) hc->list1_motion_val[1];
    l1ref0 = &hc->list1_ref_index [0][4*mb_x];
    l1ref1 = &hc->list1_ref_index [1][4*mb_x];

    /* one-to-one mv scaling */
    if(IS_16X16(*mb_type)){
        int ref, mv0, mv1;

        fill_rectangle(&m->ref_cache[1][scan8[0]], 4, 4, 8, 0, 1);
        if(IS_INTRA(mb_type_col[0])){
            ref=mv0=mv1=0;
        }else{
            const int ref0 = l1ref0[0] >= 0 ? map_col_to_list0[0][l1ref0[0]]
            : map_col_to_list0[1][l1ref1[0]];
            const int scale = dist_scale_factor[ref0];
            const int16_t *mv_col = l1ref0[0] >= 0 ? l1mv0[0] : l1mv1[0];
            int mv_l0[2];
            mv_l0[0] = (scale * mv_col[0] + 128) >> 8;
            mv_l0[1] = (scale * mv_col[1] + 128) >> 8;
            ref= ref0;
            mv0= pack16to32(mv_l0[0],mv_l0[1]);
            mv1= pack16to32(mv_l0[0]-mv_col[0],mv_l0[1]-mv_col[1]);
        }
        fill_rectangle(&m->ref_cache[0][scan8[0]], 4, 4, 8, ref, 1);
        fill_rectangle(&m-> mv_cache[0][scan8[0]], 4, 4, 8, mv0, 4);
        fill_rectangle(&m-> mv_cache[1][scan8[0]], 4, 4, 8, mv1, 4);
    }else{
        for(i8=0; i8<4; i8++){
            const int x8 = i8&1;
            const int y8 = i8>>1;
            int ref0, scale;
            const int16_t (*l1mv)[2]= l1mv0;

            if(is_b8x8 && !IS_DIRECT(m->sub_mb_type[i8]))
                continue;
            m->sub_mb_type[i8] = sub_mb_type;
            fill_rectangle(&m->ref_cache[1][scan8[i8*4]], 2, 2, 8, 0, 1);
            if(IS_INTRA(mb_type_col[0])){
                fill_rectangle(&m->ref_cache[0][scan8[i8*4]], 2, 2, 8, 0, 1);
                fill_rectangle(&m-> mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);
                fill_rectangle(&m-> mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);
                continue;
            }

            ref0 = l1ref0[i8];
            if(ref0 >= 0)
                ref0 = map_col_to_list0[0][ref0 ];
            else{
                ref0 = map_col_to_list0[1][l1ref1[i8]];
                l1mv= l1mv1;
            }
            scale = dist_scale_factor[ref0];

            fill_rectangle(&m->ref_cache[0][scan8[i8*4]], 2, 2, 8, ref0, 1);
            if(IS_SUB_8X8(sub_mb_type)){
//                 const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];
				const int16_t *mv_col = l1mv[x8*3 + y8*3*4];
                int mx = (scale * mv_col[0] + 128) >> 8;
                int my = (scale * mv_col[1] + 128) >> 8;
                fill_rectangle(&m->mv_cache[0][scan8[i8*4]], 2, 2, 8, pack16to32(mx,my), 4);
                fill_rectangle(&m->mv_cache[1][scan8[i8*4]], 2, 2, 8, pack16to32(mx-mv_col[0],my-mv_col[1]), 4);
            }else
            for(i4=0; i4<4; i4++){
//                 const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];
				const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*4];
                int16_t *mv_l0 = m->mv_cache[0][scan8[i8*4+i4]];
                mv_l0[0] = (scale * mv_col[0] + 128) >> 8;
                mv_l0[1] = (scale * mv_col[1] + 128) >> 8;
                AV_WN32A(m->mv_cache[1][scan8[i8*4+i4]],
                    pack16to32(mv_l0[0]-mv_col[0],mv_l0[1]-mv_col[1]));
            }
        }
    }
}

void ff_h264_pred_direct_motion(H264Cabac_spu *hc, EDSlice_spu *s, int *mb_type){
    if(s->direct_spatial_mv_pred){
        pred_spatial_direct_motion(hc, s, mb_type);
    }else{
        pred_temp_direct_motion(hc, s, mb_type);
    }
}
