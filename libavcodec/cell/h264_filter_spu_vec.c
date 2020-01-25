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
#include <spu_mfcio.h>
#include <spu_intrinsics.h>

#include "h264_filter_spu.h"
#include "h264_decode_mb_spu.h"
// To use scan8 table
#include "h264_mc_spu.h"


int get_chroma_qp(H264Context_spu *h, int t, int qscale){
    return h->slice.chroma_qp_table[t][qscale];
}

static inline int clip(int a, int amin, int amax){
    if (a < amin)
        return amin;
    else if (a > amax)
        return amax;
    else
        return a;
}

static inline vsint16_t clip_altivec(vsint16_t a, vsint16_t amin, vsint16_t amax){
    vector unsigned short min_mask,max_mask;
    min_mask = spu_cmpgt(amin, a);
    max_mask = spu_cmpgt(a, amax);

    return spu_sel(spu_sel(a,amin,min_mask),amax,max_mask);
}

static inline vsint16_t clip_uint8_altivec(vsint16_t a){
    const vsint16_t amax = {255,255,255,255,255,255,255,255};
    const vsint16_t amin = {0, 0, 0, 0, 0, 0, 0, 0};
    vector unsigned short min_mask,max_mask;
    min_mask = spu_cmpgt(amin, a);
    max_mask = spu_cmpgt(a, amax);

    return spu_sel(spu_sel(a,amin,min_mask),amax,max_mask);
}

static  inline void h264_loop_filter_chroma(vsint16_t *pix, int alpha, int beta, int8_t *tc0){

    short a = (short) tc0[0];
    short b = (short) tc0[1];
    short c = (short) tc0[2];
    short d = (short) tc0[3];
    const vsint16_t vec_tc0 = {a,a,b,b,c,c,d,d};
    const vsint16_t vec_v0 = {0, 0, 0, 0, 0, 0, 0, 0};
    vector unsigned short mask_B0;

    mask_B0 = spu_cmpgt(vec_v0, vec_tc0);

    const vsint16_t p0 = pix[-1];
    const vsint16_t p1 = pix[-2];
    const vsint16_t q0 = pix[0];
    const vsint16_t q1 = pix[1];

    const vsint16_t v_alpha = {(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha};
    const vsint16_t v_beta = {(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta};
    const vsint16_t v_2 = {2,2,2,2,2,2,2,2};
    const vuint16_t v_3 = {3,3,3,3,3,3,3,3};
    const vsint16_t v_4 = {4,4,4,4,4,4,4,4};

    vsint16_t rp0;
    vsint16_t rq0;
    vsint16_t abs_p0mq0, abs_p1mp0, abs_q1mq0;
    vector unsigned short mask_B1, mask_tmp;
    vsint16_t i_delta;

    abs_p0mq0 = (vector signed short) spu_absd((vector unsigned char) p0,(vector unsigned char) q0);
    abs_p1mp0 = (vector signed short) spu_absd((vector unsigned char) p1,(vector unsigned char) p0);
    abs_q1mq0 = (vector signed short) spu_absd((vector unsigned char) q1,(vector unsigned char) q0);

    mask_B1  = spu_cmpgt(v_alpha, abs_p0mq0);
    mask_tmp = spu_cmpgt(v_beta, abs_p1mp0);
    mask_B1  = spu_and(mask_B1, mask_tmp);
    mask_tmp = spu_cmpgt( v_beta, abs_q1mq0);
    mask_B1  = spu_and(mask_B1, mask_tmp);


    i_delta = clip_altivec(spu_rlmaska(spu_add(spu_sl(spu_sub(q0,p0 ), (vuint16_t)v_2), spu_add(spu_sub(p1,q1),v_4)), (vsint16_t)-v_3), -vec_tc0, vec_tc0);

    rp0 = clip_uint8_altivec( spu_add(p0,i_delta));
    rq0 = clip_uint8_altivec( spu_sub(q0,i_delta));

    pix[-1] = spu_sel(spu_sel(p0, rp0, mask_B1), p0,mask_B0);
    pix[0]  = spu_sel(spu_sel(q0, rq0, mask_B1), q0,mask_B0);
}

static void h264_v_loop_filter_luma_c(vsint16_t *pix, int alpha, int beta, int8_t *tc0, int inc_low2high){

    short a = (short) tc0[0 + inc_low2high];
    short b = (short) tc0[1 + inc_low2high];
    const vsint16_t vec_tc0 = {a,a,a,a,b,b,b,b};
    const vsint16_t vec_v0 = {0, 0, 0, 0, 0, 0, 0, 0};
    vector unsigned short mask_B0;

    mask_B0 = spu_cmpgt(vec_v0, vec_tc0);
    const vsint16_t p0 = pix[-1];
    const vsint16_t p1 = pix[-2];
    const vsint16_t p2 = pix[-3];
    const vsint16_t q0 = pix[0];
    const vsint16_t q1 = pix[1];
    const vsint16_t q2 = pix[2];

    const vuint16_t v_alpha = {(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha};
    const vuint16_t v_beta = {(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta};

    const vuint16_t v_1 = {1,1,1,1,1,1,1,1};
    const vuint16_t v_2 = {2,2,2,2,2,2,2,2};
    const vuint16_t v_3 = {3,3,3,3,3,3,3,3};
    const vsint16_t v_4 = {4,4,4,4,4,4,4,4};

    vsint16_t rp0, rp1;
    vsint16_t rq0, rq1;
    vsint16_t tc0_B2P, tc0_B2Q, rtc0;
    vuint16_t abs_p0mq0, abs_p1mp0, abs_q1mq0, abs_p2mp0, abs_q2mq0;
    vector unsigned short mask_B1, mask_B2P, mask_B2Q, mask_tmp;
    vsint16_t i_delta, i_delta2;

    abs_p0mq0 = (vector unsigned short) spu_absd((vector unsigned char) p0,(vector unsigned char) q0);
    abs_p1mp0 = (vector unsigned short) spu_absd((vector unsigned char) p1,(vector unsigned char) p0);
    abs_q1mq0 = (vector unsigned short) spu_absd((vector unsigned char) q1,(vector unsigned char) q0);
    abs_p2mp0 = (vector unsigned short) spu_absd((vector unsigned char) p2,(vector unsigned char) p0);
    abs_q2mq0 = (vector unsigned short) spu_absd((vector unsigned char) q2,(vector unsigned char) q0);

    mask_B1  = spu_cmpgt(v_alpha, abs_p0mq0);
    mask_tmp = spu_cmpgt(v_beta, abs_p1mp0);
    mask_B1  = spu_and(mask_B1, mask_tmp);
    mask_tmp = spu_cmpgt( v_beta, abs_q1mq0);
    mask_B1  = spu_and(mask_B1, mask_tmp);

    mask_B2P = spu_cmpgt(v_beta, abs_p2mp0);
    mask_B2Q = spu_cmpgt(v_beta ,abs_q2mq0);

    rp1 = spu_add(p1, clip_altivec(spu_sub(spu_rlmaska(spu_add(p2, (vector signed short) spu_avg((vector unsigned char) p0, (vector unsigned char) q0)),(vsint16_t)-v_1), p1), -vec_tc0, vec_tc0 ));
    rq1 = spu_add(q1, clip_altivec(spu_sub(spu_rlmaska(spu_add(q2, (vector signed short) spu_avg((vector unsigned char) p0, (vector unsigned char) q0)),(vsint16_t)-v_1), q1), -vec_tc0, vec_tc0 ));

    tc0_B2P = spu_add(vec_tc0, (vsint16_t) v_1);
    tc0_B2P = spu_sel(vec_tc0, tc0_B2P, mask_B2P);

    tc0_B2Q = spu_add(tc0_B2P, (vsint16_t) v_1);
    rtc0    = spu_sel(tc0_B2P, tc0_B2Q, mask_B2Q);
    i_delta2 = spu_add(spu_sub(p1,q1),v_4);
    i_delta = spu_sl(spu_sub(q0,p0 ), v_2);
    i_delta = spu_add(i_delta,i_delta2 );
    i_delta = spu_rlmaska(i_delta, (vsint16_t)-v_3);
    i_delta = clip_altivec(i_delta, -rtc0, rtc0);

    rp0 = clip_uint8_altivec( spu_add(p0,i_delta));    /* p0' */
    rq0 = clip_uint8_altivec( spu_sub(q0,i_delta));    /* q0' */

    pix[-2] = spu_sel(spu_sel(p1,spu_sel(p1,rp1,mask_B2P) ,mask_B1), p1,mask_B0);
    pix[-1] = spu_sel(spu_sel(p0, rp0, mask_B1), p0,mask_B0);
    pix[0]  = spu_sel(spu_sel(q0, rq0, mask_B1), q0,mask_B0);
    pix[1]  = spu_sel(spu_sel(q1,spu_sel(q1,rq1,mask_B2Q) ,mask_B1), q1,mask_B0);
}



static inline void h264_loop_filter_chroma_intra(vsint16_t *pix, int alpha, int beta){

    const vuint16_t p0 = (vuint16_t) pix[-1];
    const vuint16_t p1 = (vuint16_t) pix[-2];
    const vuint16_t q0 = (vuint16_t) pix[0];
    const vuint16_t q1 = (vuint16_t) pix[1];

    const vsint16_t v_alpha = {(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha,(signed short) alpha};
    const vsint16_t v_beta = {(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta,(signed short) beta};
    const vuint16_t v_2 = {2,2,2,2,2,2,2,2};

    vuint16_t rp0;
    vuint16_t rq0;
    vuint16_t abs_p0mq0, abs_p1mp0, abs_q1mq0;
    vector unsigned short mask_B0, mask_tmp;

    abs_p0mq0 = (vector unsigned short) spu_absd((vector unsigned char) p0,(vector unsigned char) q0);
    abs_p1mp0 = (vector unsigned short) spu_absd((vector unsigned char) p1,(vector unsigned char) p0);
    abs_q1mq0 = (vector unsigned short) spu_absd((vector unsigned char) q1,(vector unsigned char) q0);

    mask_B0  = spu_cmpgt(v_alpha, (vsint16_t)abs_p0mq0);
    mask_tmp = spu_cmpgt(v_beta, (vsint16_t)abs_p1mp0);
    mask_B0  = spu_and(mask_B0, mask_tmp);
    mask_tmp = spu_cmpgt( v_beta, (vsint16_t)abs_q1mq0);
    mask_B0  = spu_and(mask_B0, mask_tmp);

    rp0 = spu_add(spu_add(spu_add(p1,p0),spu_add(p1,q1)),v_2);//( 2*p1 + p0 + q1 + 2 ) >> 2;
    rp0 = spu_rlmaska(rp0, (vsint16_t)-v_2);
    rq0 = spu_add(spu_add(spu_add(q1,q0),spu_add(q1,p1)),v_2);//( 2*q1 + q0 + p1 + 2 ) >> 2;
    rq0 = spu_rlmaska(rq0, (vsint16_t)-v_2);

    pix[-1] = (vsint16_t) spu_sel(p0, rp0, mask_B0);
    pix[0]  = (vsint16_t) spu_sel(q0, rq0, mask_B0);
}
int slice_alpha_c0_offset;
int slice_beta_offset;
static void filter_mb_edgecv(vsint16_t *pix, int bS[4], int qp ) {
    int i;	
    const int index_a = qp + slice_alpha_c0_offset;
    const int alpha = (alpha_table+52)[index_a];
    const int beta  = (beta_table+52)[qp + slice_beta_offset];

    if( bS[0] < 4 ) {
        int8_t tc[4];
        for(i=0; i<4; i++)
            tc[i] = bS[i] ? tc0_table[index_a][bS[i] - 1] + 1 : 0;
        h264_loop_filter_chroma(pix, alpha, beta, tc);
    } else {
        h264_loop_filter_chroma_intra(pix, alpha, beta);
    }
}

static void filter_mb_edgeh(vsint16_t *pix, int bS[4], int qp, int inc_low2high ) {
    int i;
    const int index_a = qp + slice_alpha_c0_offset;
    const int alpha = (alpha_table+52)[index_a];
    const int beta  = (beta_table+52)[qp + slice_beta_offset];

    if( bS[0] < 4 ) {
        int8_t tc[4];
        for(i=0; i<4; i++)
            tc[i] = bS[i] ? tc0_table[index_a][bS[i] - 1] : -1;
        h264_v_loop_filter_luma_c(pix, alpha, beta, tc, inc_low2high);
    } else {

        const vuint16_t p0 = (vuint16_t) pix[-1];
        const vuint16_t p1 = (vuint16_t) pix[-2];
        const vuint16_t p2 = (vuint16_t) pix[-3];
        const vuint16_t p3 = (vuint16_t) pix[-4];
        const vuint16_t q0 = (vuint16_t) pix[0];
        const vuint16_t q1 = (vuint16_t) pix[1];
        const vuint16_t q2 = (vuint16_t) pix[2];
        const vuint16_t q3 = (vuint16_t) pix[3];

    	const vuint16_t v_alpha = {(unsigned short) alpha,(unsigned short) alpha,(unsigned short) alpha,(unsigned short) alpha,(unsigned short) alpha,(unsigned short) alpha,(unsigned short) alpha,(unsigned short) alpha};
    	const vuint16_t v_beta = {(unsigned short) beta,(unsigned short) beta,(unsigned short) beta,(unsigned short) beta,(unsigned short) beta,(unsigned short) beta,(unsigned short) beta,(unsigned short) beta};
    	const vuint16_t v_2 = {2,2,2,2,2,2,2,2};
    	const vuint16_t v_3 = {3,3,3,3,3,3,3,3};
    	const vsint16_t v_4 = {4,4,4,4,4,4,4,4};

        vuint16_t rp0_B1f, rp0_B2t, rp0_B2f, rp1_B2t, rp2_B2t;
        vuint16_t rq0_B1f, rq0_B2t, rq0_B2f, rq1_B2t, rq2_B2t;
        vuint16_t abs_p0mq0, abs_p1mp0, abs_q1mq0, abs_p2mp0, abs_q2mq0;
        vuint16_t v_alpha_2 = spu_rlmaska(v_alpha, (vsint16_t)-v_2);
        vector unsigned short mask_B0, mask_B1, mask_B2P, mask_B2Q, mask_tmp;

        v_alpha_2 = spu_add(v_alpha_2, v_2);

	abs_p0mq0 = (vector unsigned short) spu_absd((vector unsigned char) p0,(vector unsigned char) q0);
    	abs_p1mp0 = (vector unsigned short) spu_absd((vector unsigned char) p1,(vector unsigned char) p0);
    	abs_q1mq0 = (vector unsigned short) spu_absd((vector unsigned char) q1,(vector unsigned char) q0);
        abs_p2mp0 = (vector unsigned short) spu_absd((vector unsigned char) p2,(vector unsigned char) p0);
        abs_q2mq0 = (vector unsigned short) spu_absd((vector unsigned char) q2,(vector unsigned char) q0);

	mask_B0  = spu_cmpgt(v_alpha, abs_p0mq0);
	mask_tmp = spu_cmpgt(v_beta, abs_p1mp0);
	mask_B0  = spu_and(mask_B0, mask_tmp);
	mask_tmp = spu_cmpgt( v_beta, abs_q1mq0);
	mask_B0  = spu_and(mask_B0, mask_tmp);

        mask_B1  = spu_cmpgt(v_alpha_2, abs_p0mq0);
        mask_B2P = spu_cmpgt(v_beta,abs_p2mp0);
        mask_B2Q = spu_cmpgt(v_beta ,abs_q2mq0);

        rp0_B2t = spu_rlmaska(spu_add(spu_add(spu_add(spu_add(p2,p1),spu_add(p1,p0)),spu_add(spu_add(p0,q0),spu_add(q0,q1))),(vuint16_t)v_4),(vsint16_t) -v_3);
        		//( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3;
        rp1_B2t = spu_rlmaska(spu_add(spu_add(spu_add(p2,p1),spu_add(q0,p0)),v_2),(vsint16_t)-v_2);//( p2 + p1 + p0 + q0 + 2 ) >> 2;
        rp2_B2t = spu_rlmaska(spu_add(spu_add(spu_add(spu_add(p3,p3),spu_add(p2,p2)),spu_add(spu_add(p2,p1),spu_add(q0,p0))),(vuint16_t)v_4),(vsint16_t)-v_3);
        		//( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3;
        rq0_B2t = spu_rlmaska(spu_add(spu_add(spu_add(spu_add(p1,p0),spu_add(p0,q0)),spu_add(spu_add(q0,q1),spu_add(q1,q2))),(vuint16_t)v_4),(vsint16_t)-v_3);

        		//( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3;
        rq1_B2t = spu_rlmaska(spu_add(spu_add(spu_add(p0,q0),spu_add(q1,q2)),v_2),(vsint16_t)-v_2);//( p0 + q0 + q1 + q2 + 2 ) >> 2;
        rq2_B2t = spu_rlmaska(spu_add(spu_add(spu_add(spu_add(q3,q3),spu_add(q2,q2)),spu_add(spu_add(q2,q1),spu_add(q0,p0))),(vuint16_t)v_4),(vsint16_t)-v_3);
        		//( 2*q3 + 3*q2 + q1 + q0 + p0 + 4 ) >> 3;
        rp0_B1f =
        rp0_B2f = spu_rlmaska(spu_add(spu_add(spu_add(p1,p0),spu_add(p1,q1)),v_2),(vsint16_t)-v_2);//( 2*p1 + p0 + q1 + 2 ) >> 2;
        rq0_B1f =
        rq0_B2f = spu_rlmaska(spu_add(spu_add(spu_add(q1,q0),spu_add(q1,p1)),v_2),(vsint16_t)-v_2);//( 2*q1 + q0 + p1 + 2 ) >> 2;

        pix[-1] = (vsint16_t) spu_sel(p0, spu_sel(rp0_B1f, spu_sel(rp0_B2f, rp0_B2t, mask_B2P), mask_B1), mask_B0);
        pix[-2] = (vsint16_t) spu_sel(p1, spu_sel(p1, spu_sel(p1, rp1_B2t, mask_B2P), mask_B1), mask_B0);
        pix[-3] = (vsint16_t) spu_sel(p2, spu_sel(p2, spu_sel(p2, rp2_B2t, mask_B2P), mask_B1), mask_B0);
        pix[0] = (vsint16_t) spu_sel(q0, spu_sel(rq0_B1f, spu_sel(rq0_B2f, rq0_B2t, mask_B2Q), mask_B1), mask_B0);
        pix[1] = (vsint16_t) spu_sel(q1, spu_sel(q1, spu_sel(q1, rq1_B2t,mask_B2Q), mask_B1), mask_B0);
        pix[2] = (vsint16_t) spu_sel(q2, spu_sel(q2, spu_sel(q2, rq2_B2t,mask_B2Q), mask_B1), mask_B0);
    }
}

// This function gets bS and qp for luma and chroma before the filter
void calculate_bS_qp(H264Context_spu *h){
	H264mb* mb = &h->mb;
	H264slice* slice = h->slice;
    int dir;
    const int mvy_limit = 4;
    /* FIXME: A given frame may occupy more than one position in
     * the reference list. So ref2frm should be populated with
     * frame numbers, not indices. */

	int (*ref2frm)[64] = slice->ref2frm;
	int mb_x = mb->mb_x;
	int mb_y = mb->mb_y;
	int mb_type =mb->mb_type;
    /* dir : 0 -> vertical edge, 1 -> horizontal edge */
    for( dir = 0; dir < 2; dir++ ){
        int edge;
		const int mbm_type = dir == 0 ? mb->mb_type_xy_n1 : mb->mb_type_top;
        const int8_t qscale_mbm = dir == 0 ? mb->qscale_mbxy_n1 : mb->qscale_mbxy_top;

        // how often to recheck mv-based bS when iterating between edges
        const int mask_edge = (mb_type & (MB_TYPE_16x16 | (MB_TYPE_16x8 << dir))) ? 3 :(mb_type & (MB_TYPE_8x16 >> dir)) ? 1 : 0;
        // how often to recheck mv-based bS when iterating along each edge
        const int mask_par0 = mb_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir));

		h->edges[dir] = (mb_type & (MB_TYPE_16x16|MB_TYPE_SKIP)) == (MB_TYPE_16x16|MB_TYPE_SKIP) ? 1 : 4;

		if ((dir==0 && mb_x==0) || (dir==1 && mb_y==0))
			h->start[dir] =1;
		else
			h->start[dir] =0;

        /* Calculate bS */
        for( edge = h->start[dir]; edge < h->edges[dir]; edge++ ) {
            /* mbn_xy: neighbor macroblock */
            const int mbn_type = edge > 0 ? mb_type : mbm_type;
            const int8_t qscale_mbn_xy = edge > 0 ? mb->qscale_mbxy : qscale_mbm;
			int* bS = h->bS[dir][edge];

            if( (edge&1) && IS_8x8DCT(mb_type) ){
                bS[0] = bS[1] = bS[2] = bS[3] = 0; //extra code due to decoupling
                continue;
            }
            if( IS_INTRA(mb_type) ||
                IS_INTRA(mbn_type) ) {
                int value;
                if (edge == 0) {
					value = 4;
				} else {
					value = 3;
				}
                bS[0] = bS[1] = bS[2] = bS[3] = value;
            } else {
                int i, l;
                int mv_done;

                if( edge & mask_edge ) {
					bS[0] = bS[1] = bS[2] = bS[3] = 0;
                    mv_done = 1;
                }
                else if( mask_par0 && (edge || (mbn_type & (MB_TYPE_16x16 | (MB_TYPE_8x16 >> dir)))) ) {
                    int b_idx= 8 + 4 + edge * (dir ? 8:1);
                    int bn_idx= b_idx - (dir ? 8:1);
                    int v = 0;

                    for( l = 0; !v && l < 1 + (slice->slice_type_nos == FF_B_TYPE); l++ ) {
                        v |= ref2frm[mb->ref_cache[l][b_idx]+2] != ref2frm[mb->ref_cache[l][bn_idx]+2] ||
                             FFABS(mb->mv_cache[l][b_idx][0] - mb->mv_cache[l][bn_idx][0] ) >= 4 ||
                             FFABS( mb->mv_cache[l][b_idx][1] - mb->mv_cache[l][bn_idx][1] ) >= mvy_limit;
                    }
                    bS[0] = bS[1] = bS[2] = bS[3] = v;

					mv_done = 1;
                }
                else
                    mv_done = 0;

                for( i = 0; i < 4; i++ ) {
                    int x = dir == 0 ? edge : i;
                    int y = dir == 0 ? i    : edge;
                    int b_idx= 8 + 4 + x + 8*y;
                    int bn_idx= b_idx - (dir ? 8:1);

                    if( mb->non_zero_count_cache[b_idx] != 0 ||
                        mb->non_zero_count_cache[bn_idx] != 0 ) {
                        bS[i] = 2;
                    }
                    else if(!mv_done)
                    {
                        bS[i] = 0;
                        for( l = 0; l < 1 + (slice->slice_type == B_TYPE); l++ ) {
                            if( ref2frm[mb->ref_cache[l][b_idx]+2] != ref2frm[mb->ref_cache[l][bn_idx]+2] ||
                                FFABS( mb->mv_cache[l][b_idx][0] - mb->mv_cache[l][bn_idx][0] ) >= 4 ||
                                FFABS( mb->mv_cache[l][b_idx][1] - mb->mv_cache[l][bn_idx][1] ) >= mvy_limit ) {
                                bS[i] = 1;
                                break;
                            }
                        }
                    }
                }

                if(bS[0]+bS[1]+bS[2]+bS[3] == 0)
                    continue;
            }

            /* Filter edge */
            // Do not use s->qscale as luma quantizer because it has not the same
            // value in IPCM macroblocks.
            h->qp[dir][edge] = ( mb->qscale_mbxy + qscale_mbn_xy + 1 ) >> 1;
            h->chroma_qp[0][dir][edge] = ( mb->chroma_qp[0] + get_chroma_qp(h, 0, qscale_mbn_xy ) + 1 ) >> 1;

			h->chroma_qp[1][dir][edge] = ( mb->chroma_qp[1] + get_chroma_qp(h, 1, qscale_mbn_xy ) + 1 ) >> 1;
        }
		slice_alpha_c0_offset=slice->slice_alpha_c0_offset;
		slice_beta_offset= slice->slice_beta_offset;
    }
}


#define VEC_TRANSPOSE_8(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,merge_h,merge_l) \
    b0 = spu_shuffle( a0, a4, merge_h); \
    b1 = spu_shuffle( a0, a4, merge_l ); \
    b2 = spu_shuffle( a1, a5, merge_h ); \
    b3 = spu_shuffle( a1, a5, merge_l ); \
    b4 = spu_shuffle( a2, a6, merge_h ); \
    b5 = spu_shuffle( a2, a6, merge_l ); \
    b6 = spu_shuffle( a3, a7, merge_h ); \
    b7 = spu_shuffle( a3, a7, merge_l ); \
    a0 = spu_shuffle( b0, b4, merge_h ); \
    a1 = spu_shuffle( b0, b4, merge_l ); \
    a2 = spu_shuffle( b1, b5, merge_h ); \
    a3 = spu_shuffle( b1, b5, merge_l ); \
    a4 = spu_shuffle( b2, b6, merge_h ); \
    a5 = spu_shuffle( b2, b6, merge_l); \
    a6 = spu_shuffle( b3, b7, merge_h ); \
    a7 = spu_shuffle( b3, b7, merge_l ); \
    b0 = spu_shuffle( a0, a4, merge_h ); \
    b1 = spu_shuffle( a0, a4, merge_l ); \
    b2 = spu_shuffle( a1, a5, merge_h ); \
    b3 = spu_shuffle( a1, a5, merge_l); \
    b4 = spu_shuffle( a2, a6, merge_h ); \
    b5 = spu_shuffle( a2, a6, merge_l ); \
    b6 = spu_shuffle( a3, a7, merge_h ); \
    b7 = spu_shuffle( a3, a7, merge_l )

void filter_mb_spu(vsint16_t *img_y, vsint16_t *img_cb, vsint16_t *img_cr, unsigned int linesize, unsigned int uvlinesize, int edges[2], int bS[2][4][4], int qp[2][4], int chroma_qp[2][2][4], int start[2]){

    int dir,x;
    vsint16_t o_vec_img_y[(16+8)*2];
    vsint16_t t_vec_img_y[(16+8)*2];
    vsint16_t *vec_img_y_o = o_vec_img_y;
    vsint16_t *vec_img_y_t = t_vec_img_y;

    vsint16_t o_vec_img_cb[8+8+4];
    vsint16_t t_vec_img_cb[8+8];
    vsint16_t *vec_img_cb_o = &o_vec_img_cb[2];
    vsint16_t *vec_img_cb_t = t_vec_img_cb;

    vsint16_t o_vec_img_cr[8+8+4];
    vsint16_t t_vec_img_cr[8+8];
    vsint16_t *vec_img_cr_o = &o_vec_img_cr[2];
    vsint16_t *vec_img_cr_t = t_vec_img_cr;

    vuint8_t *pvec_tmp;

    const vuint8_t patt_high = {16,  0, 17,  1, 18,  2, 19,  3, 20,  4, 21,  5, 22,  6, 23,  7};
    const vuint8_t patt_low  = {16,  8, 17,  9, 18, 10, 19, 11, 20, 12, 21, 13, 22, 14, 23, 15};
    const vuint8_t patt_unpack={ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    const vuint8_t patt_pack_hw={0,  1,  2,  3,  4,  5,  6,  7, 17, 19, 21, 23, 25, 27, 29, 31};
    const vuint8_t patt_pack_chroma_aligned={0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F,
                                             0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
    const vuint8_t patt_pack_chroma_unaligned={0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                               0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F};
    const vuint8_t v_0  	   = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const vuint8_t mergehu16 = {0x00,0x01,0x10,0x11,0x02,0x03,0x12,0x13,0x04,0x05,0x14,0x15,0x06,0x07,0x16,0x17};
    const vuint8_t mergelu16 = {0x08,0x09,0x18,0x19,0x0A,0x0B,0x1A,0x1B,0x0C,0x0D,0x1C,0x1D,0x0E,0x0F,0x1E,0x1F};
    vuint8_t store_chroma, store_chroma_n1, load_chroma, load_chroma_n1;
    int mb_xy_n1;
    const int unalign_chroma = (unsigned int) img_cb & 15;

    if(unalign_chroma==0){
        load_chroma = patt_high;
        load_chroma_n1 = patt_low;  // for load chroma mb_x-1
        store_chroma = patt_pack_chroma_aligned;
        store_chroma_n1 = patt_pack_chroma_unaligned;  // for store chroma mb_x-1
        mb_xy_n1 = 1;   //  si no hay desalineamineto se necesita el bloque anterior para filtrar horizontalmente
    }
    else{
        load_chroma = patt_low;
        load_chroma_n1 = patt_high; // for load mb_x-1
        store_chroma = patt_pack_chroma_unaligned;
        store_chroma_n1 = patt_pack_chroma_aligned;    // for store chroma mb_x-1
        mb_xy_n1 = 0;   //  si hay desalineamineto 8 no se necesita el bloque anterior
    }

    /* dir : 0 -> vertical edge, 1 -> horizontal edge */

    // LOAD MB_X -1

    for (x = 0; x < 16; x++){  //Unpack Memory to 8 positions vector
        vec_img_y_o[x] = (vsint16_t) spu_shuffle((vuint8_t) img_y[x*linesize - 1], v_0 , patt_low);
    }

    for (x = 0; x < 8; x++){  //Unpack Memory to 8 positions vector
	vec_img_cb_o[x] = (vsint16_t) spu_shuffle((vuint8_t)img_cb[x*uvlinesize - mb_xy_n1], v_0 , load_chroma_n1);
	vec_img_cr_o[x] = (vsint16_t) spu_shuffle((vuint8_t)img_cr[x*uvlinesize - mb_xy_n1], v_0 , load_chroma_n1);
    }

    VEC_TRANSPOSE_8(vec_img_y_o[0], vec_img_y_o[1], vec_img_y_o[2], vec_img_y_o[3], vec_img_y_o[4], vec_img_y_o[5], vec_img_y_o[6], vec_img_y_o[7], vec_img_y_t[0], vec_img_y_t[1], vec_img_y_t[2], vec_img_y_t[3], vec_img_y_t[4], vec_img_y_t[5], vec_img_y_t[6], vec_img_y_t[7],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_o[ 8], vec_img_y_o[ 9], vec_img_y_o[10], vec_img_y_o[11], vec_img_y_o[12], vec_img_y_o[13], vec_img_y_o[14], vec_img_y_o[15], vec_img_y_t[24], vec_img_y_t[25], vec_img_y_t[26], vec_img_y_t[27], vec_img_y_t[28], vec_img_y_t[29], vec_img_y_t[30], vec_img_y_t[31],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cb_o[0], vec_img_cb_o[1], vec_img_cb_o[2], vec_img_cb_o[3], vec_img_cb_o[4], vec_img_cb_o[5], vec_img_cb_o[6], vec_img_cb_o[7], vec_img_cb_t[0], vec_img_cb_t[1], vec_img_cb_t[2], vec_img_cb_t[3], vec_img_cb_t[4], vec_img_cb_t[5], vec_img_cb_t[6], vec_img_cb_t[7],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cr_o[0], vec_img_cr_o[1], vec_img_cr_o[2], vec_img_cr_o[3], vec_img_cr_o[4], vec_img_cr_o[5], vec_img_cr_o[6], vec_img_cr_o[7], vec_img_cr_t[0], vec_img_cr_t[1], vec_img_cr_t[2], vec_img_cr_t[3], vec_img_cr_t[4], vec_img_cr_t[5], vec_img_cr_t[6], vec_img_cr_t[7],mergehu16, mergelu16);

    vec_img_y_t  = &vec_img_y_t[8];
    vec_img_y_o  = &vec_img_y_o[8];
    vec_img_cb_t = &vec_img_cb_t[8];
    vec_img_cb_o = &vec_img_cb_o[10];
    vec_img_cr_t = &vec_img_cr_t[8];
    vec_img_cr_o = &vec_img_cr_o[10];

    //LOAD CURRENT MB
    for (x = 0; x < 16; x++){  //Unpack Memory to 8 positions vector
        pvec_tmp  	  = (vuint8_t *) &img_y[x*linesize];
	vec_img_y_o[x]    = (vsint16_t) spu_shuffle(*pvec_tmp, v_0 , patt_high);
	vec_img_y_o[x+24] = (vsint16_t) spu_shuffle(*pvec_tmp, v_0 , patt_low);
    }

    for (x = 0; x < 8; x++){  //Unpack Memory to 8 positions vector
	vec_img_cb_o[x] = (vsint16_t) spu_shuffle((vuint8_t) img_cb[x*uvlinesize], v_0 , load_chroma);
	vec_img_cr_o[x] = (vsint16_t) spu_shuffle((vuint8_t) img_cr[x*uvlinesize], v_0 , load_chroma);
    }

    //TRANSPOSE MATRIX

    VEC_TRANSPOSE_8(vec_img_y_o[0], vec_img_y_o[1], vec_img_y_o[2], vec_img_y_o[3], vec_img_y_o[4], vec_img_y_o[5], vec_img_y_o[6], vec_img_y_o[7], vec_img_y_t[0], vec_img_y_t[1], vec_img_y_t[2], vec_img_y_t[3], vec_img_y_t[4], vec_img_y_t[5], vec_img_y_t[6], vec_img_y_t[7],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_o[ 8], vec_img_y_o[ 9], vec_img_y_o[10], vec_img_y_o[11], vec_img_y_o[12], vec_img_y_o[13], vec_img_y_o[14], vec_img_y_o[15], vec_img_y_t[24], vec_img_y_t[25], vec_img_y_t[26], vec_img_y_t[27], vec_img_y_t[28], vec_img_y_t[29], vec_img_y_t[30], vec_img_y_t[31],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_o[24], vec_img_y_o[25], vec_img_y_o[26], vec_img_y_o[27], vec_img_y_o[28], vec_img_y_o[29], vec_img_y_o[30], vec_img_y_o[31], vec_img_y_t[ 8], vec_img_y_t[ 9], vec_img_y_t[10], vec_img_y_t[11], vec_img_y_t[12], vec_img_y_t[13], vec_img_y_t[14], vec_img_y_t[15],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_o[32], vec_img_y_o[33], vec_img_y_o[34], vec_img_y_o[35], vec_img_y_o[36], vec_img_y_o[37], vec_img_y_o[38], vec_img_y_o[39], vec_img_y_t[32], vec_img_y_t[33], vec_img_y_t[34], vec_img_y_t[35], vec_img_y_t[36], vec_img_y_t[37], vec_img_y_t[38], vec_img_y_t[39],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cb_o[0], vec_img_cb_o[1], vec_img_cb_o[2], vec_img_cb_o[3], vec_img_cb_o[4], vec_img_cb_o[5], vec_img_cb_o[6], vec_img_cb_o[7], vec_img_cb_t[0], vec_img_cb_t[1], vec_img_cb_t[2], vec_img_cb_t[3], vec_img_cb_t[4], vec_img_cb_t[5], vec_img_cb_t[6], vec_img_cb_t[7],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cr_o[0], vec_img_cr_o[1], vec_img_cr_o[2], vec_img_cr_o[3], vec_img_cr_o[4], vec_img_cr_o[5], vec_img_cr_o[6], vec_img_cr_o[7], vec_img_cr_t[0], vec_img_cr_t[1], vec_img_cr_t[2], vec_img_cr_t[3], vec_img_cr_t[4], vec_img_cr_t[5], vec_img_cr_t[6], vec_img_cr_t[7],mergehu16, mergelu16);

    //PROCESS
    dir = 0;
    {
        int edge;
        for( edge = start[dir]; edge < edges[dir]; edge++ ) {
            if(bS[dir][edge][0]+bS[dir][edge][1]+bS[dir][edge][2]+bS[dir][edge][3] != 0)
            {
            	filter_mb_edgeh( &vec_img_y_t[4*edge   ], bS[dir][edge], qp[dir][edge],0);//low
            	filter_mb_edgeh( &vec_img_y_t[4*edge+24], bS[dir][edge], qp[dir][edge],2);//high

                if( (edge&1) == 0 ) {
                    filter_mb_edgecv( &vec_img_cb_t[2*edge], bS[dir][edge], chroma_qp[0][dir][edge] );
                    filter_mb_edgecv( &vec_img_cr_t[2*edge], bS[dir][edge], chroma_qp[1][dir][edge] );
                }
            }
        }
    }

    //SAVE MB_X -1 RESULTS

    VEC_TRANSPOSE_8(vec_img_y_t[-8], vec_img_y_t[-7], vec_img_y_t[-6], vec_img_y_t[-5], vec_img_y_t[-4], vec_img_y_t[-3], vec_img_y_t[-2], vec_img_y_t[-1], vec_img_y_o[-8], vec_img_y_o[-7], vec_img_y_o[-6], vec_img_y_o[-5], vec_img_y_o[-4], vec_img_y_o[-3], vec_img_y_o[-2], vec_img_y_o[-1],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_t[16], vec_img_y_t[17], vec_img_y_t[18], vec_img_y_t[19], vec_img_y_t[20], vec_img_y_t[21], vec_img_y_t[22], vec_img_y_t[23], vec_img_y_o[16], vec_img_y_o[17], vec_img_y_o[18], vec_img_y_o[19], vec_img_y_o[20], vec_img_y_o[21], vec_img_y_o[22], vec_img_y_o[23],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cb_t[ -8], vec_img_cb_t[-7], vec_img_cb_t[-6], vec_img_cb_t[-5], vec_img_cb_t[-4], vec_img_cb_t[-3], vec_img_cb_t[-2], vec_img_cb_t[-1], vec_img_cb_o[-10], vec_img_cb_o[-9], vec_img_cb_o[-8], vec_img_cb_o[-7], vec_img_cb_o[-6], vec_img_cb_o[-5], vec_img_cb_o[-4], vec_img_cb_o[-3],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cr_t[ -8], vec_img_cr_t[-7], vec_img_cr_t[-6], vec_img_cr_t[-5], vec_img_cr_t[-4], vec_img_cr_t[-3], vec_img_cr_t[-2], vec_img_cr_t[-1], vec_img_cr_o[-10], vec_img_cr_o[-9], vec_img_cr_o[-8], vec_img_cr_o[-7], vec_img_cr_o[-6], vec_img_cr_o[-5], vec_img_cr_o[-4], vec_img_cr_o[-3],mergehu16, mergelu16);

    for (x = 0; x < 8; x++){  //pack Memory to 8 positions vector ERROR - No check for writing out of the memory
    	img_y[x*linesize - 1] = spu_shuffle(img_y[x*linesize - 1], vec_img_y_o[-8+x], patt_pack_hw);
    }

    for (x = 0; x < 8; x++){  //pack Memory to 8 positions vector ERROR - No check for writing out of the memory
    	img_y[(x+8)*linesize - 1] = spu_shuffle(img_y[(x+8)*linesize - 1], vec_img_y_o[16+x], patt_pack_hw);
    }

    for (x = 0; x < 8; x++){  //pack Memory to 8 positions vector ERROR - No check for writing out of the memory
    	img_cb[x*uvlinesize - mb_xy_n1] = spu_shuffle(img_cb[x*uvlinesize - mb_xy_n1], vec_img_cb_o[-10+x], store_chroma_n1);
    	img_cr[x*uvlinesize - mb_xy_n1] = spu_shuffle(img_cr[x*uvlinesize - mb_xy_n1], vec_img_cr_o[-10+x], store_chroma_n1);
    }

    //TRANSPOSE MATRIX

    VEC_TRANSPOSE_8(vec_img_y_t[ 0], vec_img_y_t[ 1], vec_img_y_t[ 2], vec_img_y_t[ 3], vec_img_y_t[ 4], vec_img_y_t[ 5], vec_img_y_t[ 6], vec_img_y_t[ 7], vec_img_y_o[ 0], vec_img_y_o[ 1], vec_img_y_o[ 2], vec_img_y_o[ 3], vec_img_y_o[ 4], vec_img_y_o[ 5], vec_img_y_o[ 6], vec_img_y_o[ 7],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_t[ 8], vec_img_y_t[ 9], vec_img_y_t[10], vec_img_y_t[11], vec_img_y_t[12], vec_img_y_t[13], vec_img_y_t[14], vec_img_y_t[15], vec_img_y_o[24], vec_img_y_o[25], vec_img_y_o[26], vec_img_y_o[27], vec_img_y_o[28], vec_img_y_o[29], vec_img_y_o[30], vec_img_y_o[31],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_t[24], vec_img_y_t[25], vec_img_y_t[26], vec_img_y_t[27], vec_img_y_t[28], vec_img_y_t[29], vec_img_y_t[30], vec_img_y_t[31], vec_img_y_o[ 8], vec_img_y_o[ 9], vec_img_y_o[10], vec_img_y_o[11], vec_img_y_o[12], vec_img_y_o[13], vec_img_y_o[14], vec_img_y_o[15],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_y_t[32], vec_img_y_t[33], vec_img_y_t[34], vec_img_y_t[35], vec_img_y_t[36], vec_img_y_t[37], vec_img_y_t[38], vec_img_y_t[39], vec_img_y_o[32], vec_img_y_o[33], vec_img_y_o[34], vec_img_y_o[35], vec_img_y_o[36], vec_img_y_o[37], vec_img_y_o[38], vec_img_y_o[39],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cb_t[0], vec_img_cb_t[1], vec_img_cb_t[2], vec_img_cb_t[3], vec_img_cb_t[4], vec_img_cb_t[5], vec_img_cb_t[6], vec_img_cb_t[7], vec_img_cb_o[0], vec_img_cb_o[1], vec_img_cb_o[2], vec_img_cb_o[3], vec_img_cb_o[4], vec_img_cb_o[5], vec_img_cb_o[6], vec_img_cb_o[7],mergehu16, mergelu16);

    VEC_TRANSPOSE_8(vec_img_cr_t[0], vec_img_cr_t[1], vec_img_cr_t[2], vec_img_cr_t[3], vec_img_cr_t[4], vec_img_cr_t[5], vec_img_cr_t[6], vec_img_cr_t[7], vec_img_cr_o[0], vec_img_cr_o[1], vec_img_cr_o[2], vec_img_cr_o[3], vec_img_cr_o[4], vec_img_cr_o[5], vec_img_cr_o[6], vec_img_cr_o[7],mergehu16, mergelu16);


    //LOAD MB_Y - 1
    for (x = -4; x < 0; x++){  //Unpack Memory to 8 positions vector
	vec_img_y_o[x]    = (vsint16_t) spu_shuffle((vuint8_t) img_y[x*linesize], v_0 , patt_high);
	vec_img_y_o[x+24] = (vsint16_t) spu_shuffle((vuint8_t) img_y[x*linesize], v_0 , patt_low);
    }

    for (x = -2; x < 0; x++){  //Unpack Memory to 8 positions vector
	vec_img_cb_o[x] = (vsint16_t) spu_shuffle((vuint8_t) img_cb[x*uvlinesize], v_0 , load_chroma);
	vec_img_cr_o[x] = (vsint16_t) spu_shuffle((vuint8_t) img_cr[x*uvlinesize], v_0 , load_chroma);
    }

    //PROCESS
    dir = 1;
    {
        int edge;
        for( edge = start[dir]; edge < edges[dir]; edge++ ) {
            if(bS[dir][edge][0]+bS[dir][edge][1]+bS[dir][edge][2]+bS[dir][edge][3] != 0)
            {
            	filter_mb_edgeh( &vec_img_y_o[4*edge   ], bS[dir][edge], qp[dir][edge],0);//low
            	filter_mb_edgeh( &vec_img_y_o[4*edge+24], bS[dir][edge], qp[dir][edge],2);//high
            	if( (edge&1) == 0 ) {
            	    filter_mb_edgecv( &vec_img_cb_o[2*edge], bS[dir][edge], chroma_qp[0][dir][edge] );
                    filter_mb_edgecv( &vec_img_cr_o[2*edge], bS[dir][edge], chroma_qp[1][dir][edge] );
            	}
            }
        }

        for (x = -3; x < 16; x++){  //pack Memory to 8 positions vector ERROR - No check for writing out of the memory
    	    img_y[x*linesize] = spu_shuffle(vec_img_y_o[x], vec_img_y_o[x+24], patt_unpack);
        }

        for (x = -1; x < 8; x++){  //pack Memory to 8 positions vector ERROR - No check for writing out of the memory
            img_cb[x*uvlinesize] = spu_shuffle(img_cb[x*uvlinesize], vec_img_cb_o[x], store_chroma);
            img_cr[x*uvlinesize] = spu_shuffle(img_cr[x*uvlinesize], vec_img_cr_o[x], store_chroma);
        }
    }
}
