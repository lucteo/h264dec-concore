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

#ifndef DSPUTIL_CELL_H
#define DSPUTIL_CELL_H

#include "types_spu.h"

typedef struct DSPContext_spu {

	void (*h264_v_loop_filter_luma)(uint8_t *pix/*align 16*/, int stride, int alpha, int beta, int8_t *tc0);
    void (*h264_h_loop_filter_luma)(uint8_t *pix/*align 4 */, int stride, int alpha, int beta, int8_t *tc0);
    /* v/h_loop_filter_luma_intra: align 16 */
    void (*h264_v_loop_filter_luma_intra)(uint8_t *pix, int stride, int alpha, int beta);
    void (*h264_h_loop_filter_luma_intra)(uint8_t *pix, int stride, int alpha, int beta);
    void (*h264_v_loop_filter_chroma)(uint8_t *pix/*align 8*/, int stride, int alpha, int beta, int8_t *tc0);
    void (*h264_h_loop_filter_chroma)(uint8_t *pix/*align 4*/, int stride, int alpha, int beta, int8_t *tc0);
    void (*h264_v_loop_filter_chroma_intra)(uint8_t *pix/*align 8*/, int stride, int alpha, int beta);
    void (*h264_h_loop_filter_chroma_intra)(uint8_t *pix/*align 8*/, int stride, int alpha, int beta);

	qpel_mc_func put_h264_qpel_pixels_tab[3][16];
	qpel_mc_func avg_h264_qpel_pixels_tab[3][16];

	h264_chroma_mc_func put_h264_chroma_pixels_tab[3];
	h264_chroma_mc_func avg_h264_chroma_pixels_tab[3];

	h264_idct_func h264_idct_add[2];

	h264_weight_func weight_h264_pixels_tab[10];
	h264_biweight_func biweight_h264_pixels_tab[10];

} DSPContext_spu;


void dsputil_h264_init_cell(DSPContext_spu* c);

#endif
