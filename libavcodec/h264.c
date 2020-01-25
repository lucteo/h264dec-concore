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

#include "config.h"
#include "h264.h"
#include "h264_misc.h"
#include <math.h>

H264Context *get_h264dec_context(const char *file_name, int ifile, int ofile, int width, int height, h264_options *opts){
    int i;
    const int mb_height = (height + 15) / 16;
    const int mb_width  = (width  + 15) / 16;
    const int mb_stride = ((mb_width+1)/16 + 1) *16; //align mb_stride to 16

    ff_init_cabac_states();

    H264Context *h= av_mallocz(sizeof(H264Context));

    start_timer(h, TOTAL);
    h->file_name = file_name;
    h->profile = opts->profile;
    for (i=0; i<PROFILE_STAGES; i++)
        h->total_time[i]=0;

    h->ifile=ifile;
    h->ofile =ofile;

    h->verbose =opts->verbose;
    h->no_mbd =opts->no_mbd;
    h->static_3d =opts->static_3d;
    h->pipe_bufs = opts->pipe_bufs;
    h->slice_bufs = opts->slice_bufs;
    h->ompss_rt =opts->ompss_rt;

    h->ed_ppe_threads =0;
    if (opts->ppe_ed){
        h->ed_ppe_threads = (opts->threads >opts->ppe_ed)? opts->ppe_ed :opts->threads;
    }

    h->threads = opts->threads - h->ed_ppe_threads;
    h->smt = opts->smt;
    if (h->smt){
        h->threads *= 2;
    }

    h->num_frames = opts->numframes;

    h->frame_width = width;
    h->frame_height = height;

    while ((width/2) %STRIDE_ALIGN)
        width+=STRIDE_ALIGN;
    h->width = width;
    h->height = mb_height*16;

    h->mb_height = mb_height;
    h->mb_width = mb_width;
    h->mb_stride = mb_stride;
    h->b4_stride = mb_width*4 + 1;
    h->b_stride = mb_width*4;

    h->smb_width = opts->smb_size[0];
    h->smb_height = opts->smb_size[1] < h->smb_width ?  opts->smb_size[1]  : h->smb_width;
    h->smbc = getSuperMBContext(h, h->smb_width, h->smb_height);

    h->wave_order = opts->wave_order;

    h->pipe_bufs = opts->pipe_bufs;

    h->max_dpb_cnt = DPB_SIZE + opts->pipe_bufs;
    h->free_dpb_cnt = h->max_dpb_cnt;
    h->dpb = av_mallocz (h->max_dpb_cnt* sizeof (DecodedPicture));


    h->free_sb_cnt = h->threads*opts->slice_bufs + (h->no_mbd != 0) ;  //one extra to overlap some latency of signaling/freeing slicebuffers in entropy only mode
    h->sb_size = h->free_sb_cnt;
    h->sb = av_mallocz(h->sb_size* sizeof(SliceBufferEntry));

    h->rl_q.size = FFMAX(1, FFMIN( (h->height-3 - 512)/16, h->mb_width/2)) +1;
    h->rl_q.free = h->rl_q.size -1;
    h->rl_q.ready=0;
    h->rl_q.fi = h->rl_q.fo= 0;
    h->rl_q.queue = av_malloc(h->rl_q.size* sizeof(RingLineEntry*));
    for (i=0; i<h->rl_q.size; i++){
        if( posix_memalign((void**)&h->rl_q.queue[i],64,sizeof(RingLineEntry)))
            h->rl_q.queue[i]=NULL;
        h->rl_q.queue[i]->top = av_malloc(h->mb_width*sizeof(TopBorder));
    }

    h->rl_q.queue[0]->prev_line = h->rl_q.queue[h->rl_q.size-1];
    for (i=1; i<h->rl_q.size; i++){
        h->rl_q.queue[i]->prev_line = h->rl_q.queue[i-1];
    }

    if( HAVE_MMX | HAVE_ALTIVEC| HAVE_NEON ){
        for(i=0; i<16; i++){
            #define T(x) (x>>2) | ((x<<2) & 0xF)
            h->zigzag_scan[i] = T(zigzag_scan[i]);
            #undef T
        }
        for(i=0; i<64; i++){
            #define T(x) (x>>3) | ((x&7)<<3)
            h->zigzag_scan8x8[i]       = T(ff_zigzag_direct[i]);
            #undef T
        }
    }else{
        memcpy(h->zigzag_scan, zigzag_scan, 16*sizeof(uint8_t));
        memcpy(h->zigzag_scan8x8, ff_zigzag_direct, 64*sizeof(uint8_t));
    }

    pthread_mutex_init(&h->smb_lock, NULL);
    pthread_mutex_init(&h->sdl_lock, NULL);
    pthread_cond_init(&h->sdl_cond, NULL);

    ///pthread initialization
    pthread_mutex_init(&h->ilock, NULL);
    pthread_cond_init(&h->icond, NULL);
    pthread_mutex_init(&h->slock, NULL);
    pthread_cond_init(&h->scond, NULL);
    pthread_mutex_init(&h->tlock, NULL);
    pthread_cond_init(&h->tcond, NULL);
    pthread_mutex_init(&h->tdlock, NULL);
    pthread_cond_init(&h->tdcond, NULL);
    h->start =!opts->numamap; //default dont wait for start signal
    h->statmbd = opts->statmbd;
    h->rl_side_touch= opts->numamap;
    h->touch_start=0;
    h->setaff =opts->statsched;
    h->init_threads=0;

    pthread_mutex_init(&h->task_lock, NULL);
    pthread_cond_init(&h->task_cond, NULL);
    for (i=0; i<STAGES; i++){
        pthread_mutex_init (&h->lock[i], NULL);
        pthread_cond_init (&h->cond[i], NULL);
        pthread_mutex_init (&h->sb_q[i].lock, NULL);
        pthread_cond_init (&h->sb_q[i].cond, NULL);
        h->sb_q[i].size = h->free_sb_cnt; //change to num threads later
        h->sb_q[i].queue = av_malloc(h->free_sb_cnt* sizeof(SliceBufferEntry*));
        h->sb_q[i].cnt = h->sb_q[i].fi = h->sb_q[i].fo =0;
    }

#if HAVE_LIBSDL2
    h->sdlq.size=2;
    h->sdlq.ready=2;
    h->sdlq.queue = av_malloc(2* sizeof(SDL_Texture*));
    pthread_mutex_init (&h->sdlq.sdl_lock, NULL);
    pthread_cond_init (&h->sdlq.sdl_cond, NULL);
#endif

    h->display=opts->display;
    h->fullscreen=opts->fullscreen;

    if (opts->framerate > 0){
      h->controlfps=1;
      h->framedelay = (int64_t) (opts->framedelay*1000000000);
      h->frametime  = (int64_t) (1000000000/opts->framerate);
      struct timespec curtime;
      clock_gettime(CLOCK_REALTIME, &curtime);
      h->nexttime = curtime.tv_sec * 1000000000 + curtime.tv_nsec;

    }

    return h;
}


void free_h264dec_context(H264Context *h) {
    int i;

    for(i=0; i<h->max_dpb_cnt; i++)
        free_dp(&h->dpb[i]);
    av_free (h->dpb);

    for(i=0; i<h->sb_size; i++){
        if (h->sb[i].initialized){
            free_sb_entry(&h->sb[i]);
        }
    }
    av_freep(&h->sb);

    for (i=0; i<h->rl_q.size; i++){
        av_freep(&h->rl_q.queue[i]->top);
        av_freep(&h->rl_q.queue[i]);
    }
    av_freep(&h->rl_q.queue);

    ///pthread cleanup
    pthread_mutex_destroy (&h->task_lock);
    pthread_cond_destroy (&h->task_cond);
    for (i=0; i<STAGES; i++){
        pthread_mutex_destroy (&h->lock[i]);
        pthread_cond_destroy (&h->cond[i]);

        pthread_mutex_destroy (&h->sb_q[i].lock);
        pthread_cond_destroy (&h->sb_q[i].cond);
        av_freep( &h->sb_q[i].queue);
    }
    pthread_mutex_destroy (&h->slock);
    pthread_cond_destroy (&h->scond);
    pthread_mutex_destroy (&h->ilock);
    pthread_cond_destroy (&h->icond);

    pthread_mutex_destroy(&h->smb_lock);
    pthread_mutex_destroy (&h->sdl_lock);
    pthread_cond_destroy (&h->sdl_cond);
#if HAVE_LIBSDL2
    av_free(h->sdlq.queue);
    pthread_mutex_destroy (&h->sdlq.sdl_lock);
    pthread_cond_destroy (&h->sdlq.sdl_cond);
#endif

    stop_timer(h, TOTAL);
    if (h->threads==0){
        for (i=0; i<PROFILE_STAGES; i++)
            h->total_time[i] /= h->num_frames;
        double others = h->total_time[TOTAL];
        for (i=1; i<PROFILE_STAGES; i++)
            others-=h->total_time[i];
        if (h->profile == 1){
            printf("\n[FRAME %.3fms] [FRONT %.3fms] [ENTROPY %.3fms] [MBREC %.3fms] [OTHERS %.3fms]\n", h->total_time[TOTAL], h->total_time[FRONT], h->total_time[ED], h->total_time[REC], others);
        }else if (h->profile ==2){
            printf("\n[FRAME %.3fms] [FRONT %.3fms] [ENTROPY %.3fms] [PRED  %.3fms] [OTHERS %.3fms]\n", h->total_time[TOTAL], h->total_time[FRONT], h->total_time[ED],h->total_time[REC], others);
        }
    }

    av_free(h);
}