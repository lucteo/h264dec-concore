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


#ifndef H264_MC_H
#define H264_MC_H

#include "dsputil.h"
#include "h264_types.h"

void hl_motion(MBRecContext *d, MBRecState *mrs, H264Slice *s, H264Mb *m, uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,
					qpel_mc_func (*qpix_put)[16], h264_chroma_mc_func (*chroma_put),
					qpel_mc_func (*qpix_avg)[16], h264_chroma_mc_func (*chroma_avg),
					h264_weight_func *weight_op, h264_biweight_func *weight_avg);

#endif
