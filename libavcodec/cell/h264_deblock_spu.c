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

#include "h264_deblock_spu.h"
#include "h264_decode_mb_spu.h"

extern int print_debug;

static void filter_mb_edgev( H264Context_spu *h, uint8_t *pix, int stride, int16_t bS[4], int qp ) {
	H264slice *s= h->s;
    const int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
        tc[0] = tc0_table[index_a][bS[0]];
        tc[1] = tc0_table[index_a][bS[1]];
        tc[2] = tc0_table[index_a][bS[2]];
        tc[3] = tc0_table[index_a][bS[3]];
		
        h->dsp.h264_h_loop_filter_luma(pix, stride, alpha, beta, tc);
    } else {
        h->dsp.h264_h_loop_filter_luma_intra(pix, stride, alpha, beta);
    }
}

static void filter_mb_edgecv( H264Context_spu *h, uint8_t *pix, int stride, int16_t bS[4], int qp ) {
	H264slice *s= h->s;
    const int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
	if (alpha ==0 || beta == 0) return;
	
    if( bS[0] < 4 ) {
        int8_t tc[4];
		
        tc[0] = tc0_table[index_a][bS[0]]+1;
        tc[1] = tc0_table[index_a][bS[1]]+1;
        tc[2] = tc0_table[index_a][bS[2]]+1;
        tc[3] = tc0_table[index_a][bS[3]]+1;
		
		h->dsp.h264_h_loop_filter_chroma(pix, stride, alpha, beta, tc);
    } else {
        h->dsp.h264_h_loop_filter_chroma_intra(pix, stride, alpha, beta);
    }
}

static void filter_mb_edgeh( H264Context_spu *h, uint8_t *pix, int stride, int16_t bS[4], int qp ) {
	H264slice *s= h->s;
    const int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
		
        tc[0] = tc0_table[index_a][bS[0]];
        tc[1] = tc0_table[index_a][bS[1]];
        tc[2] = tc0_table[index_a][bS[2]];
        tc[3] = tc0_table[index_a][bS[3]];
		
        h->dsp.h264_v_loop_filter_luma(pix, stride, alpha, beta, tc);
    } else {
        h->dsp.h264_v_loop_filter_luma_intra(pix, stride, alpha, beta);
    }
}

static void filter_mb_edgech( H264Context_spu *h, uint8_t *pix, int stride, int16_t bS[4], int qp ) {
	H264slice *s= h->s;
    const int index_a = qp + s->slice_alpha_c0_offset;
    const int alpha = alpha_table[index_a];
    const int beta  = beta_table[qp + s->slice_beta_offset];
    if (alpha ==0 || beta == 0) return;

    if( bS[0] < 4 ) {
        int8_t tc[4];
		
		tc[0] = tc0_table[index_a][bS[0]]+1;
        tc[1] = tc0_table[index_a][bS[1]]+1;
        tc[2] = tc0_table[index_a][bS[2]]+1;
        tc[3] = tc0_table[index_a][bS[3]]+1;
		
        h->dsp.h264_v_loop_filter_chroma(pix, stride, alpha, beta, tc);
    } else {
        h->dsp.h264_v_loop_filter_chroma_intra(pix, stride, alpha, beta);
    }
}

static void filter_mb_dir(H264Context_spu *h, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize, int dir) {
    H264Mb *mb = h->mb;
	H264slice *s = h->s;
	const int qp_xy= mb->qscale_mb_xy;
    const int qp_dir = dir == 0 ? mb->qscale_left_mb_xy : mb->qscale_top_mb_xy;
	const int mbm_type = dir == 0 ? mb->left_type : mb->top_type;
	const int mb_type = mb->mb_type;
	int edge;
	const int edges = mb->edges[dir];
    //int (*ref2frm)[64] = s->ref2frm;

//     int start;//= h->slice_table[mbm_xy] == 0xFFFF ? 1 : 0;
// 
//     const int edges = (mb_type & (MB_TYPE_16x16|MB_TYPE_SKIP))
//                               == (MB_TYPE_16x16|MB_TYPE_SKIP) ? 1 : 4;
//     // how often to recheck mv-based bS when iterating between edges
//     const int mask_edge = (mb_type & (MB_TYPE_16x16 | (MB_TYPE_16x8 << dir))) ? 3 :
//                           (mb_type & (MB_TYPE_8x16 >> dir)) ? 1 : 0;
//     // how often to recheck mv-based bS when iterating along each edge
//     const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));

// 	if ((dir==0 && mb_x==0) || (dir==1 && mb_y==0))
// 		start =1;
// 	else
// 		start =0;
// 
//     /* Calculate bS */
//     for( edge = start; edge < edges; edge++ ) {
// 		const int mbn_type = edge > 0 ? mb_type : mbm_type;
// 		const int8_t qscale_mbn_xy = edge > 0 ? mb->qscale_mbxy : qscale_mbm;
//         int (*ref2frmn)[64] = ref2frm;//edge > 0 ? ref2frm : ref2frmm;
//         int16_t bS[4];
//         int qp;
// 
//         if( (edge&1) && IS_8x8DCT(mb_type) )
//             continue;
// 
//         if( IS_INTRA(mb_type) ||
//             IS_INTRA(mbn_type) ) {
//             int value;
// 
//             if (edge == 0) {
//                 value = 4;
//             } else {
//                 value = 3;
//             }
//             bS[0] = bS[1] = bS[2] = bS[3] = value;
//         } else {
//             int i, l;
//             int mv_done;
// 
//             if( edge & mask_edge ) {
// 
//                 bS[0] = bS[1] = bS[2] = bS[3] = 0;
//                 mv_done = 1;
//             }
//             else if( mask_par0 && (edge || (mbn_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {
//                 int b_idx= 8 + 4 + edge * (dir ? 8:1);
//                 int bn_idx= b_idx - (dir ? 8:1);
//                 int v = 0;
// 
// 				for( l = 0; !v && l < 1 + (s->slice_type_nos == FF_B_TYPE); l++ ) {
//                     v |= ref2frm[l][mb->ref_cache[l][b_idx]] != ref2frmn[l][mb->ref_cache[l][bn_idx]] ||
//                          FFABS( mb->mv_cache[l][b_idx][0] - mb->mv_cache[l][bn_idx][0] ) >= 4 ||
//                          FFABS( mb->mv_cache[l][b_idx][1] - mb->mv_cache[l][bn_idx][1] ) >= mvy_limit;
//                 }
//                 bS[0] = bS[1] = bS[2] = bS[3] = v;
// 
//                 mv_done = 1;
//             }
//             else
//                 mv_done = 0;
// 
// 			for( i = 0; i < 4; i++ ) {
//                 int x = dir == 0 ? edge : i;
//                 int y = dir == 0 ? i    : edge;
//                 int b_idx= 8 + 4 + x + 8*y;
//                 int bn_idx= b_idx - (dir ? 8:1);
// 
//                 if( mb->non_zero_count_cache[b_idx] |
//                     mb->non_zero_count_cache[bn_idx] ) {
//                     bS[i] = 2;
//                 }
//                 else if(!mv_done)
//                 {
//                     bS[i] = 0;
//                     for( l = 0; l < 1 + (s->slice_type_nos == FF_B_TYPE); l++ ) {
//                         if( ref2frm[l][mb->ref_cache[l][b_idx]] != ref2frmn[l][mb->ref_cache[l][bn_idx]] ||
//                             FFABS( mb->mv_cache[l][b_idx][0] - mb->mv_cache[l][bn_idx][0] ) >= 4 ||
//                             FFABS( mb->mv_cache[l][b_idx][1] - mb->mv_cache[l][bn_idx][1] ) >= mvy_limit ) {
//                             bS[i] = 1;
//                             break;
//                         }
//                     }
//                 }
//             }
// 
//             if(bS[0]+bS[1]+bS[2]+bS[3] == 0)
//                 continue;
//         }
// 		qp = ( mb->qscale_mbxy + qscale_mbn_xy + 1 ) >> 1;

    if(mbm_type){
        int16_t* bS=mb->bS[dir][0];
        /* Filter edge */
        // Do not use s->qscale as luma quantizer because it has not the same
        // value in IPCM macroblocks.
        if(bS[0]+bS[1]+bS[2]+bS[3]){
            int qp = ( qp_xy + qp_dir + 1 ) >> 1;
            if( dir == 0 ) {
                filter_mb_edgev(h, &img_y[0], linesize, bS, qp);
                {
                    int qp= ( get_chroma_qp(s, 0, qp_xy) + get_chroma_qp( s, 0, qp_dir) + 1 ) >> 1;
                    filter_mb_edgecv(h, &img_cb[0], uvlinesize, bS, qp);
                    filter_mb_edgecv(h, &img_cr[0], uvlinesize, bS, qp);
                }
            } else {
                filter_mb_edgeh(h, &img_y[0], linesize, bS, qp);
                {
                    int qp= ( get_chroma_qp(s, 0, qp_xy) + get_chroma_qp( s, 0, qp_dir) + 1 ) >> 1;
                    filter_mb_edgech(h, &img_cb[0], uvlinesize, bS, qp);
                    filter_mb_edgech(h, &img_cr[0], uvlinesize, bS, qp);
                }
            }
        }
    }

    for( edge = 1; edge < edges; edge++ ) {
        int16_t* bS=mb->bS[dir][edge];
        int qp = qp_xy;

        if( IS_8x8DCT(mb_type & (edge<<24)) ) // (edge&1) && IS_8x8DCT(mb_type)
            continue;

        /* Filter edge */
        // Do not use s->qscale as luma quantizer because it has not the same
        // value in IPCM macroblocks.

        if(bS[0]+bS[1]+bS[2]+bS[3] == 0)
            continue;

		if( dir == 0 ) {
            filter_mb_edgev( h, &img_y[4*edge], linesize, bS, qp );
            if( (edge&1) == 0 ) {
                filter_mb_edgecv( h, &img_cb[2*edge], uvlinesize, bS, get_chroma_qp( s, 0, qp_xy ) );
                filter_mb_edgecv( h, &img_cr[2*edge], uvlinesize, bS, get_chroma_qp( s, 1, qp_xy ) );
            }
        } else {
            filter_mb_edgeh( h, &img_y[4*edge*linesize], linesize, bS, qp );
            if( (edge&1) == 0 ) {
                filter_mb_edgech( h, &img_cb[2*edge*uvlinesize], uvlinesize, bS, get_chroma_qp( s, 0, qp_xy ) );
                filter_mb_edgech( h, &img_cr[2*edge*uvlinesize], uvlinesize, bS, get_chroma_qp( s, 1, qp_xy ) );
            }
        }
    }
}

void filter_mb( H264Context_spu *h, uint8_t *img_y, uint8_t *img_cb, uint8_t *img_cr, unsigned int linesize, unsigned int uvlinesize) {
    filter_mb_dir(h, img_y, img_cb, img_cr, linesize, uvlinesize, 0);
    filter_mb_dir(h, img_y, img_cb, img_cr, linesize, uvlinesize, 1);
}
