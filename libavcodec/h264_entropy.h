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
#ifndef H264_CABAC_H
#define H264_CABAC_H

#include "h264_types.h"
#include "cabac.h"

/**
 * decodes a CABAC coded macroblock
 * @return 0 if OK, AC_ERROR / DC_ERROR / MV_ERROR if an error is noticed
 */

int ff_h264_decode_mb_cabac(EntropyContext *ec, H264Slice *s, CABACContext *c);
void ff_h264_init_cabac_states(EntropyContext *ec, H264Slice *s, CABACContext *c);

int init_entropy_buf(EntropyContext *ec, H264Slice *s, int line);
EntropyContext * get_entropy_context(H264Context *h);
void init_dequant_tables(H264Slice *s, EntropyContext *ec);
void free_entropy_context(EntropyContext *ec);

#endif
