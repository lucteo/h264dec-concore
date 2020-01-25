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


#ifndef H264_MISC_H
#define H264_MISC_H

#include "avcodec.h"
#include "h264_types.h"

void start_timer(H264Context *h, int stage);
void stop_timer(H264Context *h, int stage);

void init_sb_entry(H264Context *h, SliceBufferEntry *sbe);
void free_sb_entry(SliceBufferEntry *sb);
SliceBufferEntry *get_sb_entry(H264Context *h);
void release_sb_entry(H264Context *h, SliceBufferEntry *sb);

DecodedPicture *get_dpb_entry(H264Context *h, H264Slice *s);
void release_dpb_entry(H264Context *h, DecodedPicture *pic, int mode);

void draw_edges(MBRecContext *d, H264Slice *s, int line);

int ff_init_slice(NalContext *n, H264Slice *s);
void free_picture(PictureInfo *pic);
void free_dp(DecodedPicture *pic);

void av_start_timer();
int copyEDtoH264Slice(H264Slice *ms, H264Slice *es);
void print_report(int frame_number, int dropped_frames, uint64_t video_size, int is_last_report, int verbose);

int ff_alloc_picture_info(NalContext *n, H264Slice *s, PictureInfo *pic);
DecodedPicture *output_frame(H264Context *h, OutputContext *oc, DecodedPicture *pic, int fd, int frame_width, int frame_height);
OutputContext *get_output_context(H264Context *h);
void free_output_context(OutputContext *oc);

void freeSuperMBContext(SuperMBContext *smbc);
SuperMBContext *getSuperMBContext(H264Context *h, int smb_width, int smb_height);
void release_smbc(H264Context *h, SuperMBContext *smbc);
SuperMBContext * acquire_smbc(H264Context *h );

#if HAVE_LIBSDL2
void signal_sdl_exit(H264Context *h);
void *sdl_thread(void *arg);
SDLContext *get_SDL_context(H264Context *h);
void free_SDL_context(SDLContext *sdlc);
#endif

/**
* gets the chroma qp.
*/
static inline int get_chroma_qp(H264Slice *s, int t, int qscale){
    return s->pps.chroma_qp_table[t][qscale];
}

#endif
