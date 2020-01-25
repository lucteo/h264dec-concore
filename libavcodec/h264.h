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

#ifndef H264_H
#define H264_H

#include "h264_entropy.h"
#include "h264_data.h"
#include "h264_mc.h"
#include "h264_misc.h"
#include "h264_dsp.h"
#include "h264_pred.h"
#include "h264_parser.h"
#include "h264_nal.h"
#include "h264_rec.h"
#include "h264_deblock.h"
#include "h264_types.h"

typedef struct h264_options{
    int statsched;
    int statmbd;
    int numamap;
    int no_mbd;
    int numframes;
    int display;
    int fullscreen;
    int verbose;
    int ppe_ed;         // only useful for Cell
    int profile;
    int threads;
    int smb_size[2];    // only useful for OmpSs
    int wave_order;
    int static_3d;
    int ompss_rt;
    int pipe_bufs;
    int slice_bufs;
    int smt;
    double framerate;
    double framedelay;
}h264_options;

int h264_decode_cell(H264Context *h);
int h264_decode_cell_seq(H264Context *h);

int h264_decode_ompss(H264Context *h);

int h264_decode_pthread(H264Context *h);
int h264_decode_seq(H264Context *h);


H264Context *get_h264dec_context(const char *file_name, int ifile, int ofile, int frame_width, int frame_height, h264_options *opts);
void free_h264dec_context(H264Context *h);


#endif /* AVCODEC_H264_H */
