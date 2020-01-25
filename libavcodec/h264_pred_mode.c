/*
 * H.26L/H.264/AVC/JVT/14496-10/... direct mb/block decoding
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
 * H.264 / AVC / MPEG4 part10 direct mb/block decoding.
 * @author Michael Niedermayer <michaelni@gmx.at>
 * @author Chi Ching Chi <chi.c.chi@tu-berlin.de>
 */

#include "dsputil.h"
#include "avcodec.h"
#include "h264_data.h"
#include "h264.h"
#include "rectangle.h"

//#undef NDEBUG
#include <assert.h>

static const uint8_t left_block_options[4][16]={
    {0,1,2,3,7,10,8,11,7+0*8, 7+1*8, 7+2*8, 7+3*8, 2+0*8, 2+3*8, 2+1*8, 2+2*8},
    {2,2,3,3,8,11,8,11,7+2*8, 7+2*8, 7+3*8, 7+3*8, 2+1*8, 2+2*8, 2+1*8, 2+2*8},
    {0,0,1,1,7,10,7,10,7+0*8, 7+0*8, 7+1*8, 7+1*8, 2+0*8, 2+3*8, 2+0*8, 2+3*8},
    {0,2,0,2,7,10,7,10,7+0*8, 7+2*8, 7+0*8, 7+2*8, 2+0*8, 2+3*8, 2+0*8, 2+3*8}
};


// static void check_cache_copy(MBRecContext *mrc, H264Slice *s, H264Mb *m){
//     for (int list=0; list<2; list++){
//         for (int i=0; i<40; i++){
//             assert (m->ref_cache[list][i] == m->ref_cache_copy[list][i]);
//             assert (mrs->mv_cache[list][i][0] == mrs->mv_cache_copy[list][i][0]);
//             assert (mrs->mv_cache[list][i][1] == mrs->mv_cache_copy[list][i][1]);
//         }
//     }
// }

// static void check_cache_copy2(MBRecContext *mrc, H264Slice *s, H264Mb *m){
//     for (int list=0; list<2; list++){
//         for (int i=0; i<40; i++){
//             assert (m->ref_cache[list][i] == m->ref_cache_copy2[list][i]);
//             assert (mrs->mv_cache[list][i][0] == mrs->mv_cache_copy2[list][i][0]);
//             assert (mrs->mv_cache[list][i][1] == mrs->mv_cache_copy2[list][i][1]);
//         }
//     }
// }

static void fill_decode_caches_rec(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int mb_type){
    int topleft_type, top_type, topright_type, left_type;
    const uint8_t * left_block= left_block_options[0];
    const int mb_x = m->mb_x;
    int i;

    mrs->top_type  = mrs->mb_type_top[mb_x  ];
    mrs->left_type = mrs->mb_type    [mb_x-1];

    topleft_type = mrs->mb_type_top[mb_x-1];
    top_type     = mrs->mb_type_top[mb_x  ];
    topright_type= mrs->mb_type_top[mb_x+1];
    left_type    = mrs->mb_type    [mb_x-1];

    int type_mask= s->pps.constrained_intra_pred ? 1 : -1;

    if(!IS_SKIP(mb_type)){
//         memset(mrc->non_zero_count_cache, 0, sizeof(mrc->non_zero_count_cache));
        AV_COPY32(&mrs->non_zero_count_cache[4+8*1], &m->non_zero_count[ 0]);
        AV_COPY32(&mrs->non_zero_count_cache[4+8*2], &m->non_zero_count[ 4]);
        AV_COPY32(&mrs->non_zero_count_cache[4+8*3], &m->non_zero_count[ 8]);
        AV_COPY32(&mrs->non_zero_count_cache[4+8*4], &m->non_zero_count[12]);

        for (int i=0; i<2; i++) {
            mrs->non_zero_count_cache[8*1 + 8*i + 1] = m->non_zero_count[16 + i*2   ];
            mrs->non_zero_count_cache[8*1 + 8*i + 2] = m->non_zero_count[16 + i*2 +1];
            mrs->non_zero_count_cache[8*4 + 8*i + 1] = m->non_zero_count[20 + i*2   ];
            mrs->non_zero_count_cache[8*4 + 8*i + 2] = m->non_zero_count[20 + i*2 +1];
        }

        if(IS_INTRA(mb_type)){
//             memset(mrc->intra4x4_pred_mode_cache, 0, sizeof(mrc->intra4x4_pred_mode_cache));

            mrs->topleft_samples_available=
            mrs->top_samples_available=
            mrs->left_samples_available= 0xFFFF;
            mrs->topright_samples_available= 0xEEEA;

            if(!(top_type & type_mask)){
                mrs->topleft_samples_available= 0xB3FF;
                mrs->top_samples_available= 0x33FF;
                mrs->topright_samples_available= 0x26EA;
            }

            if(!(left_type & type_mask)){
                mrs->topleft_samples_available&= 0xDF5F;
                mrs->left_samples_available&= 0x5F5F;
            }

            if(!(topleft_type & type_mask))
                mrs->topleft_samples_available&= 0x7FFF;

            if(!(topright_type & type_mask))
                mrs->topright_samples_available&= 0xFBFF;

            if(IS_INTRA4x4(mb_type)){
                if(IS_INTRA4x4(top_type)){
                    AV_COPY32(mrs->intra4x4_pred_mode_cache+4+8*0, &mrs->intra4x4_pred_mode_top[4*mb_x]);
                }else{
                    mrs->intra4x4_pred_mode_cache[4+8*0]=
                    mrs->intra4x4_pred_mode_cache[5+8*0]=
                    mrs->intra4x4_pred_mode_cache[6+8*0]=
                    mrs->intra4x4_pred_mode_cache[7+8*0]= 2 - 3*!(top_type & type_mask);
                }

                if(IS_INTRA4x4(left_type)){
#if OMPSS
                    mrs->intra4x4_pred_mode_cache[3+8*1]= m->intra4x4_pred_mode_left[0];
                    mrs->intra4x4_pred_mode_cache[3+8*2]= m->intra4x4_pred_mode_left[1];
                    mrs->intra4x4_pred_mode_cache[3+8*3]= m->intra4x4_pred_mode_left[2];
                    mrs->intra4x4_pred_mode_cache[3+8*4]= m->intra4x4_pred_mode_left[3];
#else
                    mrs->intra4x4_pred_mode_cache[3+8*1]= mrs->intra4x4_pred_mode_left[0];
                    mrs->intra4x4_pred_mode_cache[3+8*2]= mrs->intra4x4_pred_mode_left[1];
                    mrs->intra4x4_pred_mode_cache[3+8*3]= mrs->intra4x4_pred_mode_left[2];
                    mrs->intra4x4_pred_mode_cache[3+8*4]= mrs->intra4x4_pred_mode_left[3];
#endif
                }else{
                    mrs->intra4x4_pred_mode_cache[3+8*1]=
                    mrs->intra4x4_pred_mode_cache[3+8*2]=
                    mrs->intra4x4_pred_mode_cache[3+8*3]=
                    mrs->intra4x4_pred_mode_cache[3+8*4]= 2 - 3*!(left_type & type_mask);
                }
            }
        }
    }

    if(IS_INTER(mb_type) ||(IS_DIRECT(mb_type) && s->direct_spatial_mv_pred)){
        int list;

//         memset(mrs->mv_cache, 0, sizeof(mrs->mv_cache));
//         memset(mrs->ref_cache, 0, sizeof(mrs->ref_cache));

        mrs->ref_cache[0][scan8[5 ]+1] = mrs->ref_cache[0][scan8[7 ]+1] = mrs->ref_cache[0][scan8[13]+1] =
        mrs->ref_cache[1][scan8[5 ]+1] = mrs->ref_cache[1][scan8[7 ]+1] = mrs->ref_cache[1][scan8[13]+1] = PART_NOT_AVAILABLE;

        for(list=0; list<s->list_count; list++){
            if(!USES_LIST(mb_type, list)){
                continue;
            }
            assert(!(IS_DIRECT(mb_type) && !s->direct_spatial_mv_pred));

            if(USES_LIST(top_type, list)){
                const int b_xy= 4*mb_x + 3*mrc->b_stride;
                AV_COPY128(mrs->mv_cache[list][scan8[0] + 0 - 1*8], mrs->motion_val_top[list][b_xy + 0]);
                    mrs->ref_cache[list][scan8[0] + 0 - 1*8]=
                    mrs->ref_cache[list][scan8[0] + 1 - 1*8]= mrs->ref_index_top[list][4*mb_x + 2];
                    mrs->ref_cache[list][scan8[0] + 2 - 1*8]=
                    mrs->ref_cache[list][scan8[0] + 3 - 1*8]= mrs->ref_index_top[list][4*mb_x + 3];
            }else{
                AV_ZERO128(mrs->mv_cache[list][scan8[0] + 0 - 1*8]);
                AV_WN32A(&mrs->ref_cache[list][scan8[0] + 0 - 1*8], ((top_type ? LIST_NOT_USED : PART_NOT_AVAILABLE)&0xFF)*0x01010101);
            }

            if(mb_type & (MB_TYPE_16x8|MB_TYPE_8x8)){
                for(i=0; i<2; i++){
                    int cache_idx = scan8[0] - 1 + i*2*8;
                    if(USES_LIST(left_type, list)){
                        const int b_xy= 4*(mb_x-1) + 3;
                        const int b8_x= 4*(mb_x-1) + 1;
                        AV_COPY32(mrs->mv_cache[list][cache_idx  ], mrs->motion_val[list][b_xy + mrc->b_stride*left_block[0+i*2]]);
                        AV_COPY32(mrs->mv_cache[list][cache_idx+8], mrs->motion_val[list][b_xy + mrc->b_stride*left_block[1+i*2]]);
                        mrs->ref_cache[list][cache_idx  ]= mrs->ref_index[list][b8_x + (left_block[0+i*2]&~1)];
                        mrs->ref_cache[list][cache_idx+8]= mrs->ref_index[list][b8_x + (left_block[1+i*2]&~1)];
                    }else{
                        AV_ZERO32(mrs->mv_cache [list][cache_idx  ]);
                        AV_ZERO32(mrs->mv_cache [list][cache_idx+8]);
                        mrs->ref_cache[list][cache_idx  ]=
                        mrs->ref_cache[list][cache_idx+8]= (left_type ? LIST_NOT_USED : PART_NOT_AVAILABLE);
                    }
                }
            }else{
                if(USES_LIST(left_type, list)){
                    const int b_x = 4*(mb_x-1) + 3;
                    const int b8_x= 4*(mb_x-1) + 1;
                    AV_COPY32(mrs->mv_cache[list][scan8[0] - 1], mrs->motion_val[list][b_x + mrc->b_stride*left_block[0]]);
                    mrs->ref_cache[list][scan8[0] - 1]= mrs->ref_index[list][b8_x + (left_block[0]&~1)];
                }else{
                    AV_ZERO32(mrs->mv_cache [list][scan8[0] - 1]);
                    mrs->ref_cache[list][scan8[0] - 1]= left_type ? LIST_NOT_USED : PART_NOT_AVAILABLE;
                }
            }

            if(USES_LIST(topright_type, list)){
                const int b_xy= 4*(mb_x+1) + 3*mrc->b_stride;
                AV_COPY32(mrs->mv_cache[list][scan8[0] + 4 - 1*8], mrs->motion_val_top[list][b_xy]);
                mrs->ref_cache[list][scan8[0] + 4 - 1*8]= mrs->ref_index_top[list][4*(mb_x+1) + 2];
            }else{
                AV_ZERO32(mrs->mv_cache [list][scan8[0] + 4 - 1*8]);
                mrs->ref_cache[list][scan8[0] + 4 - 1*8]= topright_type ? LIST_NOT_USED : PART_NOT_AVAILABLE;
            }
            if(mrs->ref_cache[list][scan8[0] + 4 - 1*8] < 0){
                int topleft_partition= -1;
                if(USES_LIST(topleft_type, list)){
                    const int b_xy = 4*(mb_x-1) + 3 + mrc->b_stride + (topleft_partition & 2*mrc->b_stride);
                    const int b8_x= 4*(mb_x-1) + 1 + (topleft_partition & 2);
                    AV_COPY32(mrs->mv_cache[list][scan8[0] - 1 - 1*8], mrs->motion_val_top[list][b_xy]);
                    mrs->ref_cache[list][scan8[0] - 1 - 1*8]= mrs->ref_index_top[list][b8_x];
                }else{
                    AV_ZERO32(mrs->mv_cache[list][scan8[0] - 1 - 1*8]);
                    mrs->ref_cache[list][scan8[0] - 1 - 1*8]= topleft_type ? LIST_NOT_USED : PART_NOT_AVAILABLE;
                }
            }

            if((mb_type&(MB_TYPE_SKIP|MB_TYPE_DIRECT2)))
                continue;

            if(!(mb_type&(MB_TYPE_SKIP|MB_TYPE_DIRECT2))) {
                mrs->ref_cache[list][scan8[4 ]] =
                mrs->ref_cache[list][scan8[12]] = PART_NOT_AVAILABLE;
                AV_ZERO32(mrs->mv_cache [list][scan8[4 ]]);
                AV_ZERO32(mrs->mv_cache [list][scan8[12]]);
            }
        }
    }
}

static inline void write_back_motion_rec(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int mb_type){
    const int b_stride = mrc->b_stride;
    const int b_x = 4*m->mb_x; //try mb2b(8)_xy
    const int b8_x= 4*m->mb_x;
    int list;

    if(!USES_LIST(mb_type, 0))
        fill_rectangle(&mrs->ref_index[0][b8_x], 2, 2, 2, (uint8_t)LIST_NOT_USED, 1);

    for(list=0; list<s->list_count; list++){
        int y;
        int16_t (*mv_dst)[2];
        int16_t (*mv_src)[2];

        if(!USES_LIST(mb_type, list))
            continue;

        mv_dst   = &mrs->motion_val[list][b_x];
        mv_src   = &mrs->mv_cache[list][scan8[0]];
        for(y=0; y<4; y++){
            AV_COPY128(mv_dst + y*b_stride, mv_src + 8*y);
        }

        {
            int8_t *ref_index = &mrs->ref_index[list][b8_x];
            ref_index[0+0*2]= mrs->ref_cache[list][scan8[0]];
            ref_index[1+0*2]= mrs->ref_cache[list][scan8[4]];
            ref_index[0+1*2]= mrs->ref_cache[list][scan8[8]];
            ref_index[1+1*2]= mrs->ref_cache[list][scan8[12]];
        }
    }
}


/**
* checks if the top & left blocks are available if needed & changes the dc mode so it only uses the available blocks.
*/
static int check_intra4x4_pred_mode(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m){
    static const int8_t top [12]= {-1, 0,LEFT_DC_PRED,-1,-1,-1,-1,-1, 0};
    static const int8_t left[12]= { 0,-1, TOP_DC_PRED, 0,-1,-1,-1, 0,-1,DC_128_PRED};
    int i;

    if(!(mrs->top_samples_available&0x8000)){
        for(i=0; i<4; i++){
            int status= top[ mrs->intra4x4_pred_mode_cache[scan8[0] + i] ];
            if(status<0){
                av_log(AV_LOG_ERROR, "top block unavailable for requested intra4x4 mode %d at %d %d\n", status, m->mb_x, m->mb_y);
                return -1;
            } else if(status){
                mrs->intra4x4_pred_mode_cache[scan8[0] + i]= status;
            }
        }
    }

    if((mrs->left_samples_available&0x8888)!=0x8888){
        static const int mask[4]={0x8000,0x2000,0x80,0x20};
        for(i=0; i<4; i++){
            if(!(mrs->left_samples_available&mask[i])){
                int status= left[ mrs->intra4x4_pred_mode_cache[scan8[0] + 8*i] ];
                if(status<0){
                    av_log(AV_LOG_ERROR, "left block unavailable for requested intra4x4 mode %d at %d %d\n", status, m->mb_x, m->mb_y);
                    return -1;
                } else if(status){
                    mrs->intra4x4_pred_mode_cache[scan8[0] + 8*i]= status;
                }
            }
        }
    }
    return 0;
}

/**
* checks if the top & left blocks are available if needed & changes the dc mode so it only uses the available blocks.
*/
static int check_intra_pred_mode(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int mode){
    static const int8_t top [7]= {LEFT_DC_PRED8x8, 1,-1,-1};
    static const int8_t left[7]= { TOP_DC_PRED8x8,-1, 2,-1,DC_128_PRED8x8};

    if(mode > 6) {
        av_log(AV_LOG_ERROR, "out of range intra chroma pred mode at %d %d\n", m->mb_x, m->mb_y);
        return -1;
    }

    if(!(mrs->top_samples_available&0x8000)){
        mode= top[ mode ];
        if(mode<0){
            av_log(AV_LOG_ERROR, "top block unavailable for requested intra mode at %d %d\n", m->mb_x, m->mb_y);
            return -1;
        }
    }

    if((mrs->left_samples_available&0x8080) != 0x8080){
        mode= left[ mode ];
        if(mrs->left_samples_available&0x8080){ //mad cow disease mode, aka MBAFF + constrained_intra_pred
            mode= ALZHEIMER_DC_L0T_PRED8x8 + (!(mrs->left_samples_available&0x8000)) + 2*(mode == DC_128_PRED8x8);
        }
        if(mode<0){
            av_log(AV_LOG_ERROR, "left block unavailable for requested intra mode at %d %d\n", m->mb_x, m->mb_y);
            return -1;
        }
    }
    return mode;
}

/**
 * gets the predicted intra4x4 prediction mode.
 */
static inline int pred_intra_mode(MBRecContext *mrc, MBRecState *mrs, int n){
    const int index8= scan8[n];
    const int left= mrs->intra4x4_pred_mode_cache[index8 - 1];
    const int top = mrs->intra4x4_pred_mode_cache[index8 - 8];
    const int min= FFMIN(left, top);

    if(min<0) return DC_PRED;
    else      return min;
}

static void write_back_intra_pred_mode_rec(MBRecContext *mrc, MBRecState *mrs, H264Mb *m, int mb_x){
    int8_t *mode= &mrs->intra4x4_pred_mode[4*mb_x];

    AV_COPY32(mode, mrs->intra4x4_pred_mode_cache + 4 + 8*4);
#if OMPSS
    if (m->mb_x < mrc->mb_width-1){
        H264Mb *mr= m+1;
        mode = mr->intra4x4_pred_mode_left;
        mode[0]= mrs->intra4x4_pred_mode_cache[7+8*1];
        mode[1]= mrs->intra4x4_pred_mode_cache[7+8*2];
        mode[2]= mrs->intra4x4_pred_mode_cache[7+8*3];
        mode[3]= mrs->intra4x4_pred_mode_cache[7+8*4];
    }
#else
    mode = mrs->intra4x4_pred_mode_left;
    mode[0]= mrs->intra4x4_pred_mode_cache[7+8*1];
    mode[1]= mrs->intra4x4_pred_mode_cache[7+8*2];
    mode[2]= mrs->intra4x4_pred_mode_cache[7+8*3];
    mode[3]= mrs->intra4x4_pred_mode_cache[7+8*4];
#endif
}

static void pred_spatial_direct_motion_rec(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int *mb_type){
    int b4_stride = mrc->b_stride;
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
        int left_ref = mrs->ref_cache[list][scan8[0] - 1];
        int top_ref  = mrs->ref_cache[list][scan8[0] - 8];
        int refc = mrs->ref_cache[list][scan8[0] - 8 + 4];
        const int16_t *C= mrs->mv_cache[list][ scan8[0] - 8 + 4];
        if(refc == PART_NOT_AVAILABLE){
            refc = mrs->ref_cache[list][scan8[0] - 8 - 1];
            C    = mrs->mv_cache[list][scan8[0] - 8 - 1];
        }
        ref[list] = FFMIN3((unsigned)left_ref, (unsigned)top_ref, (unsigned)refc);
        if(ref[list] >= 0){
            //this is just pred_motion() but with the cases removed that cannot happen for direct blocks
            const int16_t * const A= mrs->mv_cache[list][ scan8[0] - 1 ];
            const int16_t * const B= mrs->mv_cache[list][ scan8[0] - 8 ];

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
        fill_rectangle(&mrs->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);
        fill_rectangle(&mrs->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);
        fill_rectangle(&mrs->mv_cache[0][scan8[0]], 4, 4, 8, 0, 4);
        fill_rectangle(&mrs->mv_cache[1][scan8[0]], 4, 4, 8, 0, 4);
        *mb_type= (*mb_type & ~(MB_TYPE_8x8|MB_TYPE_16x8|MB_TYPE_8x16|MB_TYPE_P1L0|MB_TYPE_P1L1))|MB_TYPE_16x16|MB_TYPE_DIRECT2;
        return;
    }

    mb_type_col[0] =
    mb_type_col[1] = mrs->list1_mb_type[mb_x];

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

    l1mv0  = (void *) &mrs->list1_motion_val[0][4*mb_x];
    l1mv1  = (void *) &mrs->list1_motion_val[1][4*mb_x];
    l1ref0 = &mrs->list1_ref_index [0][4*mb_x];
    l1ref1 = &mrs->list1_ref_index [1][4*mb_x];
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

        fill_rectangle(&mrs->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);
        fill_rectangle(&mrs->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);
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
        fill_rectangle(&mrs->mv_cache[0][scan8[0]], 4, 4, 8, a, 4);
        fill_rectangle(&mrs->mv_cache[1][scan8[0]], 4, 4, 8, b, 4);
    }else{
        int n=0;
        for(i8=0; i8<4; i8++){
            const int x8 = i8&1;
            const int y8 = i8>>1;

            if(is_b8x8 && !IS_DIRECT(m->sub_mb_type[i8]))
                continue;
            m->sub_mb_type[i8] = sub_mb_type;

            fill_rectangle(&mrs->mv_cache[0][scan8[i8*4]], 2, 2, 8, mv[0], 4);
            fill_rectangle(&mrs->mv_cache[1][scan8[i8*4]], 2, 2, 8, mv[1], 4);
            fill_rectangle(&mrs->ref_cache[0][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[0], 1);
            fill_rectangle(&mrs->ref_cache[1][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[1], 1);

            /* col_zero_flag */
            if(!IS_INTRA(mb_type_col[0]) && (l1ref0[i8] == 0 || (l1ref0[i8] < 0 && l1ref1[i8] == 0 ))
                ){
                const int16_t (*l1mv)[2]= l1ref0[i8] == 0 ? l1mv0 : l1mv1;
                if(IS_SUB_8X8(sub_mb_type)){
                    const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];
                    if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){
                        if(ref[0] == 0)
                            fill_rectangle(&mrs->mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);
                        if(ref[1] == 0)
                            fill_rectangle(&mrs->mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);
                        n+=4;
                    }
                }else{
                    int k=0;
                    for(i4=0; i4<4; i4++){
                        const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];
                        if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){
                            if(ref[0] == 0)
                                AV_ZERO32(mrs->mv_cache[0][scan8[i8*4+i4]]);
                            if(ref[1] == 0)
                                AV_ZERO32(mrs->mv_cache[1][scan8[i8*4+i4]]);
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

static void pred_temp_direct_motion_rec(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int *mb_type){
    const int mb_x = m->mb_x;
    int b4_stride = mrc->b_stride;
    int mb_type_col[2];
    const int16_t (*l1mv0)[2], (*l1mv1)[2];
    const int8_t *l1ref0, *l1ref1;
    const int is_b8x8 = IS_8X8(*mb_type);
    unsigned int sub_mb_type;
    int i8, i4;
    const int *map_col_to_list0[2] = {s->map_col_to_list0[0], s->map_col_to_list0[1]};
    const int *dist_scale_factor = s->dist_scale_factor;

    mb_type_col[0] =
    mb_type_col[1] = mrs->list1_mb_type[mb_x];

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

    l1mv0  = (void *) &mrs->list1_motion_val[0][4*mb_x];
    l1mv1  = (void *) &mrs->list1_motion_val[1][4*mb_x];
    l1ref0 = &mrs->list1_ref_index [0][4*mb_x];
    l1ref1 = &mrs->list1_ref_index [1][4*mb_x];

    /* one-to-one mv scaling */
    if(IS_16X16(*mb_type)){
        int ref, mv0, mv1;

        fill_rectangle(&mrs->ref_cache[1][scan8[0]], 4, 4, 8, 0, 1);
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
        fill_rectangle(&mrs->ref_cache[0][scan8[0]], 4, 4, 8, ref, 1);
        fill_rectangle(&mrs->mv_cache[0][scan8[0]], 4, 4, 8, mv0, 4);
        fill_rectangle(&mrs->mv_cache[1][scan8[0]], 4, 4, 8, mv1, 4);
    }else{
        for(i8=0; i8<4; i8++){
            const int x8 = i8&1;
            const int y8 = i8>>1;
            int ref0, scale;
            const int16_t (*l1mv)[2]= l1mv0;

            if(is_b8x8 && !IS_DIRECT(m->sub_mb_type[i8]))
                continue;
            m->sub_mb_type[i8] = sub_mb_type;
            fill_rectangle(&mrs->ref_cache[1][scan8[i8*4]], 2, 2, 8, 0, 1);
            if(IS_INTRA(mb_type_col[0])){
                fill_rectangle(&mrs->ref_cache[0][scan8[i8*4]], 2, 2, 8, 0, 1);
                fill_rectangle(&mrs->mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);
                fill_rectangle(&mrs->mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);
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

            fill_rectangle(&mrs->ref_cache[0][scan8[i8*4]], 2, 2, 8, ref0, 1);
            if(IS_SUB_8X8(sub_mb_type)){
                const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];
                int mx = (scale * mv_col[0] + 128) >> 8;
                int my = (scale * mv_col[1] + 128) >> 8;
                fill_rectangle(&mrs->mv_cache[0][scan8[i8*4]], 2, 2, 8, pack16to32(mx,my), 4);
                fill_rectangle(&mrs->mv_cache[1][scan8[i8*4]], 2, 2, 8, pack16to32(mx-mv_col[0],my-mv_col[1]), 4);
            }else
            for(i4=0; i4<4; i4++){
                const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];
                int16_t *mv_l0 = mrs->mv_cache[0][scan8[i8*4+i4]];
                mv_l0[0] = (scale * mv_col[0] + 128) >> 8;
                mv_l0[1] = (scale * mv_col[1] + 128) >> 8;
                AV_WN32A(mrs->mv_cache[1][scan8[i8*4+i4]],
                    pack16to32(mv_l0[0]-mv_col[0],mv_l0[1]-mv_col[1]));
            }
        }
    }
}

void ff_h264_pred_direct_motion_rec(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m, int *mb_type){
    if(s->direct_spatial_mv_pred){
        pred_spatial_direct_motion_rec(mrc, mrs, s, m, mb_type);
    }else{
        pred_temp_direct_motion_rec(mrc, mrs, s, m, mb_type);
    }
}

static inline int fetch_diagonal_mv(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, const int16_t **C, int i, int list, int part_width){
    const int topright_ref= mrs->ref_cache[list][ i - 8 + part_width ];

    if(topright_ref != PART_NOT_AVAILABLE){
        *C= mrs->mv_cache[list][ i - 8 + part_width ];
        return topright_ref;
    }else{
        *C= mrs->mv_cache[list][ i - 8 - 1 ];
        return mrs->ref_cache[list][ i - 8 - 1 ];
    }
}

/**
 * gets the predicted MV.
 * @param n the block index
 * @param part_width the width of the partition (4, 8,16) -> (1, 2, 4)
 * @param mx the x component of the predicted motion vector
 * @param my the y component of the predicted motion vector
 */
static inline void pred_motion(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, int n, int part_width, int list, int ref, int * const mx, int * const my){
    const int index8= scan8[n];
    const int top_ref=      mrs->ref_cache[list][ index8 - 8 ];
    const int left_ref=     mrs->ref_cache[list][ index8 - 1 ];
    const int16_t * const A= mrs->mv_cache[list][ index8 - 1 ];
    const int16_t * const B= mrs->mv_cache[list][ index8 - 8 ];
    const int16_t * C;
    int diagonal_ref, match_count;

    assert(part_width==1 || part_width==2 || part_width==4);

/* mv_cache
  B . . A T T T T
  U . . L . . , .
  U . . L . . . .
  U . . L . . , .
  . . . L . . . .
*/

    diagonal_ref= fetch_diagonal_mv(mrc, mrs, s, &C, index8, list, part_width);
    match_count= (diagonal_ref==ref) + (top_ref==ref) + (left_ref==ref);

    if(match_count > 1){ //most common
        *mx= mid_pred(A[0], B[0], C[0]);
        *my= mid_pred(A[1], B[1], C[1]);
    }else if(match_count==1){
        if(left_ref==ref){
            *mx= A[0];
            *my= A[1];
        }else if(top_ref==ref){
            *mx= B[0];
            *my= B[1];
        }else{
            *mx= C[0];
            *my= C[1];
        }
    }else{
        if(top_ref == PART_NOT_AVAILABLE && diagonal_ref == PART_NOT_AVAILABLE && left_ref != PART_NOT_AVAILABLE){
            *mx= A[0];
            *my= A[1];
        }else{
            *mx= mid_pred(A[0], B[0], C[0]);
            *my= mid_pred(A[1], B[1], C[1]);
        }
    }

}

/**
 * gets the directionally predicted 16x8 MV.
 * @param n the block index
 * @param mx the x component of the predicted motion vector
 * @param my the y component of the predicted motion vector
 */
static inline void pred_16x8_motion(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, int n, int list, int ref, int * const mx, int * const my){
    if(n==0){
        const int top_ref=      mrs->ref_cache[list][ scan8[0] - 8 ];
        const int16_t * const B= mrs->mv_cache[list][ scan8[0] - 8 ];

        if(top_ref == ref){
            *mx= B[0];
            *my= B[1];
            return;
        }
    }else{
        const int left_ref=     mrs->ref_cache[list][ scan8[8] - 1 ];
        const int16_t * const A= mrs->mv_cache[list][ scan8[8] - 1 ];

        if(left_ref == ref){
            *mx= A[0];
            *my= A[1];
            return;
        }
    }

    //RARE
    pred_motion(mrc, mrs, s, n, 4, list, ref, mx, my);
}

/**
 * gets the directionally predicted 8x16 MV.
 * @param n the block index
 * @param mx the x component of the predicted motion vector
 * @param my the y component of the predicted motion vector
 */
static inline void pred_8x16_motion(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, int n, int list, int ref, int * const mx, int * const my){
    if(n==0){
        const int left_ref=      mrs->ref_cache[list][ scan8[0] - 1 ];
        const int16_t * const A=  mrs->mv_cache[list][ scan8[0] - 1 ];

        if(left_ref == ref){
            *mx= A[0];
            *my= A[1];
            return;
        }
    }else{
        const int16_t * C;
        int diagonal_ref;

        diagonal_ref= fetch_diagonal_mv(mrc, mrs, s, &C, scan8[4], list, 2);
        if(diagonal_ref == ref){
            *mx= C[0];
            *my= C[1];
            return;
        }
    }

    //RARE
    pred_motion(mrc, mrs, s, n, 2, list, ref, mx, my);
}

static inline void pred_pskip_motion(MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb * m, int * const mx, int * const my){
    const int top_ref = mrs->ref_cache[0][ scan8[0] - 8 ];
    const int left_ref= mrs->ref_cache[0][ scan8[0] - 1 ];

    if(top_ref == PART_NOT_AVAILABLE || left_ref == PART_NOT_AVAILABLE
       || !( top_ref | AV_RN32A(mrs->mv_cache[0][ scan8[0] - 8 ]))
       || !(left_ref | AV_RN32A(mrs->mv_cache[0][ scan8[0] - 1 ]))){

        *mx = *my = 0;
        return;
    }

    pred_motion(mrc, mrs, s, 0, 4, 0, 0, mx, my);

    return;
}

#define ADD_MVD(list) \
{ \
    mx += m->mvd[list][mp][0]; \
    my += m->mvd[list][mp][1]; \
    mp++; \
}

int pred_motion_mb_rec (MBRecContext *mrc, MBRecState *mrs, H264Slice *s, H264Mb *m){
    int mp=0;
    int mb_type = m->mb_type;
    const int mb_x = m->mb_x;

//     mrc->m =m;

    fill_decode_caches_rec(mrc, mrs, s, m, mb_type);
    if (IS_SKIP(mb_type)){
        mb_type=0;

        if( s->slice_type_nos == FF_B_TYPE )
        {
            mb_type|= MB_TYPE_L0L1|MB_TYPE_DIRECT2|MB_TYPE_SKIP;
            ff_h264_pred_direct_motion_rec(mrc, mrs, s, m, &mb_type);
        }
        else
        {
            int mx, my;

            mb_type|= MB_TYPE_16x16|MB_TYPE_P0L0|MB_TYPE_P1L0|MB_TYPE_SKIP; //FIXME check required
            pred_pskip_motion(mrc, mrs, s, m, &mx, &my);
            fill_rectangle(&mrs->ref_cache[0][scan8[0]], 4, 4, 8, 0, 1);
            fill_rectangle(mrs->mv_cache[0][scan8[0]], 4, 4, 8, pack16to32(mx,my), 4);
        }

        write_back_motion_rec(mrc, mrs, s, m, mb_type);
        m->mb_type = mrs->mb_type[mb_x]= mb_type;
        return 0;
    }


    if (IS_INTRA_PCM(mb_type)){
        mrs->mb_type[mb_x] =  mb_type;
        return 0;
    }
    else if (IS_INTRA(mb_type)){
        int i, pred_mode;

        if( IS_INTRA4x4( mb_type ) ) {
            if ( IS_8x8DCT(mb_type) ) {
                for( i = 0; i < 16; i+=4 ) {
                    int pred = pred_intra_mode(mrc, mrs, i );
                    int mode = m->intra4x4_pred_mode[i];

                    mode = mode < 0 ?  pred : mode + ( mode >= pred );
                    fill_rectangle( &mrs->intra4x4_pred_mode_cache[ scan8[i] ], 2, 2, 8, mode, 1 );
                }
            } else {
                for( i = 0; i < 16; i++ ) {
                    int pred = pred_intra_mode(mrc, mrs, i );
                    int mode = m->intra4x4_pred_mode[i];
                    mode = mode < 0 ?  pred : mode + ( mode >= pred );
                    mrs->intra4x4_pred_mode_cache[ scan8[i] ] = mode;
                }
            }
            write_back_intra_pred_mode_rec(mrc, mrs, m, mb_x);
            if( check_intra4x4_pred_mode(mrc, mrs, s, m) < 0 ) return -1;
        } else {
            m->intra16x16_pred_mode= check_intra_pred_mode(mrc, mrs, s, m, m->intra16x16_pred_mode );
            if( m->intra16x16_pred_mode < 0 ) return -1;
        }

        pred_mode = m->chroma_pred_mode;
        pred_mode= check_intra_pred_mode( mrc, mrs, s, m, pred_mode );
        if( pred_mode < 0 ) return -1;
        m->chroma_pred_mode= pred_mode;

    }
    else if (IS_8X8(mb_type)){
        int i, j, list;

        if( s->slice_type_nos == FF_B_TYPE ) {
            if( IS_DIRECT(m->sub_mb_type[0] | m->sub_mb_type[1] |
                            m->sub_mb_type[2] | m->sub_mb_type[3]) ) {
                ff_h264_pred_direct_motion_rec(mrc, mrs, s, m, &mb_type);
                mrs->ref_cache[0][scan8[4]] =
                mrs->ref_cache[1][scan8[4]] =
                mrs->ref_cache[0][scan8[12]] =
                mrs->ref_cache[1][scan8[12]] = PART_NOT_AVAILABLE;
            }
        }

        for(list=0; list<s->list_count; list++){
            for(i=0; i<4; i++){
                if(IS_DIRECT(m->sub_mb_type[i])){
                    mrs->ref_cache[list][ scan8[4*i]   ]=mrs->ref_cache[list][ scan8[4*i]+1 ];
                    continue;
                } else {
                    mrs->ref_cache[list][ scan8[4*i]   ]=mrs->ref_cache[list][ scan8[4*i]+1 ]=
                    mrs->ref_cache[list][ scan8[4*i]+8 ]=mrs->ref_cache[list][ scan8[4*i]+9 ]= m->ref_index[list][i];

                    if(IS_DIR(m->sub_mb_type[i], 0, list) ){
                        const int sub_mb_type= m->sub_mb_type[i];
                        const int block_width= (sub_mb_type & (MB_TYPE_16x16|MB_TYPE_16x8)) ? 2 : 1;

                        int sub_partition_count = IS_SUB_8X8(sub_mb_type) ? 1 : (IS_SUB_4X4(sub_mb_type)? 4 :2);
                        for(j=0; j<sub_partition_count; j++){
                            int mx, my;
                            const int index= 4*i + block_width*j;
                            int16_t (* mv_cache)[2]= &mrs->mv_cache[list][ scan8[index]];
                            pred_motion(mrc, mrs, s, index, block_width, list, mrs->ref_cache[list][ scan8[index] ], &mx, &my);

                            ADD_MVD(list)

                            if(IS_SUB_8X8(sub_mb_type)){
                                mv_cache[ 1 ][0]=
                                mv_cache[ 8 ][0]= mv_cache[ 9 ][0]= mx;
                                mv_cache[ 1 ][1]=
                                mv_cache[ 8 ][1]= mv_cache[ 9 ][1]= my;
                            }else if(IS_SUB_8X4(sub_mb_type)){
                                mv_cache[ 1 ][0]= mx;
                                mv_cache[ 1 ][1]= my;
                            }else if(IS_SUB_4X8(sub_mb_type)){
                                mv_cache[ 8 ][0]= mx;
                                mv_cache[ 8 ][1]= my;
                            }
                            mv_cache[ 0 ][0]= mx;
                            mv_cache[ 0 ][1]= my;
                        }
                    }else{
                        fill_rectangle(mrs->mv_cache [list][ scan8[4*i] ], 2, 2, 8, 0, 4);
                    }
                }
            }
        }
    } else if( IS_DIRECT(mb_type) ) {
        mb_type &= ~MB_TYPE_16x16;  //FIXME not nice
        ff_h264_pred_direct_motion_rec(mrc, mrs, s, m, &mb_type);
    }
    else {
        int list, i;
        if(IS_16X16(mb_type)){
            for(list=0; list<s->list_count; list++){
                if(IS_DIR(mb_type, 0, list)){
                    int ref;
                    int mx,my;

                    ref = m->ref_index[list][0];
                    fill_rectangle(&mrs->ref_cache[list][ scan8[0] ], 4, 4, 8, ref, 1);
                    pred_motion(mrc, mrs, s, 0, 4, list, mrs->ref_cache[list][ scan8[0] ], &mx, &my);
                    ADD_MVD(list)
                    fill_rectangle(mrs->mv_cache[list][ scan8[0] ], 4, 4, 8, pack16to32(mx,my), 4);
                }
            }
        }
        else if(IS_16X8(mb_type)){
            for(list=0; list<s->list_count; list++){
                for(i=0; i<2; i++){
                    if(IS_DIR(mb_type, i, list)){
                        int ref;
                        int mx,my;
                        ref = m->ref_index[list][i];
                        fill_rectangle(&mrs->ref_cache[list][ scan8[0] + 16*i ], 4, 2, 8, ref, 1);

                        pred_16x8_motion(mrc, mrs, s, 8*i, list, mrs->ref_cache[list][scan8[0] + 16*i], &mx, &my);
                        ADD_MVD(list)

                        fill_rectangle(mrs->mv_cache[list][ scan8[0] + 16*i ], 4, 2, 8, pack16to32(mx,my), 4);
                    }else{
                        fill_rectangle(&mrs->ref_cache[list][ scan8[0] + 16*i ], 4, 2, 8, (LIST_NOT_USED&0xFF), 1);
                        fill_rectangle(mrs->mv_cache[list][ scan8[0] + 16*i ], 4, 2, 8, 0, 4);
                    }
                }
            }

        }else{
            assert(IS_8X16(mb_type));

            for(list=0; list<s->list_count; list++){
                for(i=0; i<2; i++){
                    if(IS_DIR(mb_type, i, list)){ //FIXME optimize
                        int ref;
                        int mx,my;
                        ref = m->ref_index[list][i];
                        fill_rectangle(&mrs->ref_cache[list][ scan8[0] + 2*i ], 2, 4, 8, ref, 1);
                        pred_8x16_motion(mrc, mrs, s, i*4, list, mrs->ref_cache[list][ scan8[0] + 2*i ], &mx, &my);
                        ADD_MVD(list)
                        fill_rectangle(mrs->mv_cache[list][ scan8[0] + 2*i ], 2, 4, 8, pack16to32(mx,my), 4);
                    }else{
                        fill_rectangle(&mrs->ref_cache[list][ scan8[0] + 2*i ], 2, 4, 8, (LIST_NOT_USED&0xFF), 1);
                        fill_rectangle(mrs->mv_cache[list][ scan8[0] + 2*i ], 2, 4, 8, 0, 4);
                    }
                }
            }
        }
    }

    if (IS_INTER(mb_type)||(IS_DIRECT(mb_type)))
        write_back_motion_rec(mrc, mrs, s, m, mb_type);
    m->mb_type = mrs->mb_type[mb_x]= mb_type;

    return 0;
}
