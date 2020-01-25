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

#include "h264_types.h"

#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#undef NDEBUG
#include <assert.h>

#if HAVE_LIBSDL2
#include <SDL2/SDL.h>
#if HAVE_LIBSDL_TTF
#include <SDL/SDL_ttf.h>
#endif
#endif

void start_timer(H264Context *h, int stage){
    clock_gettime(CLOCK_REALTIME, &h->start_time[stage]);
}

void stop_timer(H264Context *h, int stage){
    clock_gettime(CLOCK_REALTIME, &h->end_time[stage]);
    double time = (double) 1.e3*(h->end_time[stage].tv_sec - h->start_time[stage].tv_sec) + 1.e-6*(h->end_time[stage].tv_nsec - h->start_time[stage].tv_nsec);
    h->last_time [stage]  = time;
    h->total_time[stage] += time;
}

void init_sb_entry(H264Context *h, SliceBufferEntry *sbe){
    sbe->mbs = av_malloc(h->mb_width*h->mb_height* sizeof(H264Mb));
    sbe->initialized = 1;
}

void free_sb_entry(SliceBufferEntry *sbe){
    av_free(sbe->mbs);
    av_freep(&sbe->gb.raw);
    if (sbe->gb.rbsp)
        av_freep(&sbe->gb.rbsp);
    sbe->initialized = 0;
}

SliceBufferEntry *get_sb_entry(H264Context *h){
    SliceBufferEntry *sb = NULL;

    pthread_mutex_lock(&h->lock[PARSE]);
    while (h->free_sb_cnt<=0)
        pthread_cond_wait(&h->cond[PARSE], &h->lock[PARSE]);
    /* use first free picture */
    for(int i=0; i<h->sb_size; i++){
        if(h->sb[i].state==0){
            sb= &h->sb[i];
            sb->state=1;
            sb->lines_taken=0;
            sb->lines_total=h->mb_height;
            break;
        }
    }
    h->free_sb_cnt--;

    pthread_mutex_unlock(&h->lock[PARSE]);

    memset (&sb->slice, 0, sizeof(H264Slice));

    return sb;
}

void release_sb_entry(H264Context *h, SliceBufferEntry *sb){
    pthread_mutex_lock(&h->lock[PARSE]);

    sb->state = 0;
    h->free_sb_cnt++;
    pthread_cond_signal(&h->cond[PARSE]);

    pthread_mutex_unlock(&h->lock[PARSE]);
}

int init_dpb_entry(H264Context *h, DecodedPicture *pic, H264Slice *s, int width, int height){
    int i;

    s->curr_pic=pic;
    pic->poc = s->poc;
    pic->key_frame = s->key_frame;
    pic->mmco_reset = s->mmco_reset;
    pic->reference = s->nal_ref_idc? 3:1;
    pic->cpn = s->coded_pic_num;
    pic->drop =0;

    if(pic->data[0]==NULL) {
        int size[3] = {0};

        width+= EDGE_WIDTH*2;
        height+= EDGE_WIDTH*2;

        pic->linesize[0]= width;
        pic->linesize[1]=  pic->linesize[2] = width>>1;

        size[0] = width*height;
        size[1] = size[2] = width*height>>2;

        for(i=0; i<3; i++){
            pic->base[i]= av_malloc(size[i]);
        }

        pic->data[0] = pic->base[0] + (pic->linesize[0]*EDGE_WIDTH) + EDGE_WIDTH;
        pic->data[1] = pic->base[1] + (pic->linesize[1]*EDGE_WIDTH>>1) + (EDGE_WIDTH>>1);
        pic->data[2] = pic->base[2] + (pic->linesize[2]*EDGE_WIDTH>>1) + (EDGE_WIDTH>>1);
    }

    const int big_mb_num= h->mb_stride*(h->mb_height+1) + 1; //the +1 is needed so memset(,,stride*height) does not sig11
    const int mb_array_size= h->mb_stride*h->mb_height;
    const int b4_array_size= h->b4_stride*h->mb_height*4;

    if(pic->mb_type_base==NULL){
        FF_ALLOCZ_OR_GOTO(pic->mb_type_base , big_mb_num * sizeof(uint32_t), fail)
        pic->mb_type= pic->mb_type_base + h->mb_stride+1;

        for(int i=0; i<2; i++){
            FF_ALLOCZ_OR_GOTO(pic->motion_val_base[i], 2 * (b4_array_size+4)  * sizeof(int16_t), fail)
            pic->motion_val[i]= pic->motion_val_base[i]+4;
            FF_ALLOCZ_OR_GOTO(pic->ref_index[i], 4*mb_array_size * sizeof(uint8_t), fail)
        }
        FF_ALLOCZ_OR_GOTO(pic->intra4x4_pred_mode, h->mb_width*h->mb_height * 4* sizeof(int8_t), fail)
    }

    return 0;
    fail:
    return -1;
}

void free_dp(DecodedPicture *pic){
    if(pic->base[0]){
        for (int i=0; i<3; i++){
            av_free(pic->base[i]);
            pic->data[i]= NULL;
        }
    }
    if (pic->mb_type_base){
        av_free(pic->mb_type_base);
        pic->mb_type= NULL;
        for(int i=0; i<2; i++){
            av_free(pic->motion_val_base[i]);
            av_free(pic->ref_index[i]);
        }
        av_free(pic->intra4x4_pred_mode);
    }
}

DecodedPicture *get_dpb_entry(H264Context *h, H264Slice *s){
    DecodedPicture *dp = NULL;

    pthread_mutex_lock(&h->lock[REORDER2]);
//     printf("%d\n", h->free_dpb_cnt);

    while (h->free_dpb_cnt<=0){
    #if OMPSS
        assert(0);
    #endif
        pthread_cond_wait(&h->cond[REORDER2], &h->lock[REORDER2]);
    }
    /* use first free picture */
    for(int i=0; i<h->max_dpb_cnt; i++){
        if(h->dpb[i].reference==0){
            dp= &h->dpb[i];
            break;
        }
    }
    assert(dp);
    init_dpb_entry(h, dp, s, h->width, h->height);
    h->free_dpb_cnt--;
    h->acdpb_cnt++; //debug
    pthread_mutex_unlock(&h->lock[REORDER2]);

    return dp;
}

void release_dpb_entry(H264Context *h, DecodedPicture *pic, int mode){
    pthread_mutex_lock(&h->lock[REORDER2]);
    pic->reference &= ~mode;
    if (pic->reference == 0){
        h->free_dpb_cnt++;
        h->reldpb_cnt++; //debug
        pthread_cond_signal(&h->cond[REORDER2]);
    }
    pthread_mutex_unlock(&h->lock[REORDER2]);
}


/**
*   Extends the edges of a macroblock line.
*/
void draw_edges(MBRecContext *d, H264Slice *s, int line){
    int i;
    int mb_width=d->mb_width;
    int mb_height=d->mb_height;
    int last = (line+1 == mb_height);
    int lines = last?16:12;
    int linesize = d->linesize;
    int uvlinesize = d->uvlinesize;
    uint8_t *y = s->curr_pic->data[0] + 16*line*linesize;
    uint8_t *cb = s->curr_pic->data[1] + 8*line*uvlinesize;
    uint8_t *cr = s->curr_pic->data[2] + 8*line*uvlinesize;

    for (i=-4; i<lines; i++){
        memset(y + i*linesize - EDGE_WIDTH, y[i*linesize], EDGE_WIDTH);
        memset(y + i*linesize + mb_width*16, y[i*linesize +mb_width*16 -1], EDGE_WIDTH);
    }
    for (i=-2; i<lines/2; i++){
        memset(cb + i*uvlinesize - EDGE_WIDTH/2, cb[i*uvlinesize], EDGE_WIDTH/2);
        memset(cb + i*uvlinesize + mb_width*8, cb[i*uvlinesize +mb_width*8 -1], EDGE_WIDTH/2);
        memset(cr + i*uvlinesize - EDGE_WIDTH/2, cr[i*uvlinesize], EDGE_WIDTH/2);
        memset(cr + i*uvlinesize + mb_width*8, cr[i*uvlinesize +mb_width*8 -1], EDGE_WIDTH/2);
    }

    if (line==0){
        y -= EDGE_WIDTH;
        cb -= EDGE_WIDTH/2;
        cr -= EDGE_WIDTH/2;
        for (i=1; i<=21; i++){
            memcpy(y -i*linesize, y, linesize);
        }
        for (i=1; i<=9; i++){
            memcpy(cb -i*uvlinesize, cb, uvlinesize);
            memcpy(cr -i*uvlinesize, cr, uvlinesize);
        }
    }else if (last){
        y += -EDGE_WIDTH + 15*linesize;
        cb += -EDGE_WIDTH/2 + 7*uvlinesize;
        cr += -EDGE_WIDTH/2 + 7*uvlinesize;
        for (i=1; i<=21; i++){
            memcpy(y +i*linesize, y, linesize);
        }
        for (i=1; i<=9; i++){
            memcpy(cb +i*uvlinesize, cb, uvlinesize);
            memcpy(cr +i*uvlinesize, cr, uvlinesize);
        }
    }
}

static int64_t timer_start;
int64_t av_gettime(void) {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

void av_start_timer(){
    timer_start = av_gettime();
}

void print_report(int frame_number, int dropped_frames, uint64_t video_size, int is_last_report, int verbose) {
    static int64_t last_time = -1;
    static int64_t last_frame_number = 0;
    float t=0, t2=0;
    int64_t cur_time=0;

    if (!is_last_report) {
        /* display the report every 0.5 seconds */
        cur_time = av_gettime();
        if (last_time == -1) {
            last_time = cur_time;
            return;
        }
        if ((cur_time - last_time) < 500000)
            return;
        t = (cur_time-timer_start) / 1000000.0;
        t2 = (cur_time-last_time) / 1000000.0;
    }

    if (verbose){
        fprintf(stderr, "frames=%5d dropped=%5d avgfps=%3d curfps=%3d\r", frame_number, dropped_frames, (int)(frame_number/t+0.5), (int)((frame_number - last_frame_number)/t2+0.5) );
        fflush(stderr);
    }
    last_frame_number = frame_number;
    last_time = cur_time;

    if (is_last_report){
        t = (av_gettime()-timer_start) / 1000000.0;
        fprintf(stderr, "%c[2Kframe=%5d avgfps=%3d\r", 27, frame_number, (int)(frame_number/t+0.5));
        fprintf(stderr, "\n");
        fprintf(stderr, "video:%1.0fkB\n", video_size/1024.0);
        fflush(stderr);
    }
}

/* Sort B-frames into display order */
static DecodedPicture *get_reordered_picture(OutputContext *w, int flush){
    int i;
    int out_idx = 0;
    DecodedPicture *out = w->delayed_pic[0];

    if (!out)
        return NULL;

    for(i=1; w->delayed_pic[i] && !w->delayed_pic[i]->key_frame && !w->delayed_pic[i]->mmco_reset; i++){
        if(w->delayed_pic[i]->poc < out->poc){
            out = w->delayed_pic[i];
            out_idx = i;
        }
    }

    if(w->dp_cnt > MAX_DELAYED_PIC_COUNT || flush) {
        for(i=out_idx; w->delayed_pic[i]; i++)
            w->delayed_pic[i] = w->delayed_pic[i+1];
        w->dp_cnt--;
        return out;
    }
    return NULL;
}

/**
*  Remove the extra borders, and places the three parts of the image after each other.
*/
static int raw_encode(const DecodedPicture* src, int width, int height, unsigned char *dest) {
    int i, j;
/** To write entire image including extra borders*/
//  int w = src->linesize[0];
//  int h = height+64;
//  int w2 = w>>1;
//  int h2 = h>>1;
//     int data_planes=3;
//     int size = w * h + 2 *w2*h2;
//     const unsigned char* s;
//     for (i=0; i<data_planes; i++) {
//         if (i == 1) {
//             w = w2;
//             h = h2;
//         }
//         s = src->base[i];
//         for(j=0; j<h; j++) {
//             memcpy(dest, s, src->linesize[i]);
//             dest += w;
//             s += src->linesize[i];
//         }
//     }

    int w = (width*8 + 7)/8;
    int h = height;
    int w2 =((width >>1) * 8 + 7) / 8;
    int h2 = ((height+1) >>1); //not sure about +1
    int data_planes=3;
    int size = w * h + 2 *w2*h2;
    const unsigned char* s;


    for (i=0; i<data_planes; i++) {
        if (i == 1) {
            w = w2;
            h = h2;
        }
        s = src->data[i];
        for(j=0; j<h; j++) {
            memcpy(dest, s, w);
            dest += w;
            s += src->linesize[i];
        }
    }
    return size;
}

#ifdef HAVE_LIBSDL2
static SDL_Texture *get_next_texture(H264Context *h, int side){
    SDLTextureQueue *sdlq = &h->sdlq;
    SDL_Texture *texture;
    pthread_mutex_lock (&sdlq->sdl_lock);
    if (side ){ //send
        while (sdlq->ready >= sdlq->size)
            pthread_cond_wait(&sdlq->sdl_cond, &sdlq->sdl_lock);
        texture = sdlq->queue[sdlq->fi];
        sdlq->fi++; sdlq->fi %= sdlq->size;
    } else { //recv
        while (sdlq->ready <= 0 && !sdlq->exit)
            pthread_cond_wait(&sdlq->sdl_cond, &sdlq->sdl_lock);

        if (sdlq->ready == 0 && sdlq->exit){
            texture = NULL;
        }else{
            texture = sdlq->queue[sdlq->fo];
            sdlq->fo++; sdlq->fo %= sdlq->size;
        }
    }
    pthread_mutex_unlock(&sdlq->sdl_lock);

    return texture;
}

static void signal_texture(H264Context *h, int side){
    SDLTextureQueue *sdlq = &h->sdlq;
    pthread_mutex_lock (&sdlq->sdl_lock);
    if (side)
        sdlq->ready++;
    else
        sdlq->ready--;
    pthread_cond_signal(&sdlq->sdl_cond);
    pthread_mutex_unlock(&sdlq->sdl_lock);
}

void signal_sdl_exit(H264Context *h){
    SDLTextureQueue *sdlq = &h->sdlq;
    pthread_mutex_lock (&sdlq->sdl_lock);
    sdlq->exit=1;
    pthread_cond_signal(&sdlq->sdl_cond);
    pthread_mutex_unlock(&sdlq->sdl_lock);
}

static void display_frame(H264Context *h, OutputContext *w, int fd, DecodedPicture *in_picture, int frame_width, int frame_height, int dropable){
    static int64_t last_time = -1;
    int64_t cur_time;
//     SDLContext *sdlc = h->sdlc;
    uint8_t *iyuv_pixels;
    int pitch;


    if (last_time == -1){
        last_time = av_gettime();
    }


    /* do not display frames that are less than 8.125 ms apart (120fps)*/
    if (dropable){
        cur_time = av_gettime();

        if ((cur_time - last_time) < 8125)
            return;

        last_time =cur_time;
    }

    if(in_picture){

        SDL_Texture *texture= get_next_texture(h, 1);

        SDL_LockTexture( texture, NULL, (void **)&iyuv_pixels, &pitch );

        raw_encode(in_picture, frame_width, frame_height, iyuv_pixels);

        signal_texture(h, 1);
    }
}
#endif

// TODO: Parallelize the raw_encode (either split frame or over frames)
static void do_video_out(OutputContext *w, int fd, DecodedPicture *in_picture, int frame_width, int frame_height) {
    int size=0;
    //remove extra borders

    if(in_picture)
        size= raw_encode(in_picture, frame_width, frame_height, w->bit_buffer);

    if (size < 0) {
        fprintf(stderr, "Video encoding failed\n");
    }else {
        if (write(fd, w->bit_buffer, size)<0)
            fprintf(stderr, "Write frame failed\n");
    }

    w->video_size += size;
}

DecodedPicture *output_frame(H264Context *h, OutputContext *oc, DecodedPicture *pic, int fd, int frame_width, int frame_height) {
    DecodedPicture *out;

    if (pic){
        oc->delayed_pic[oc->dp_cnt++]=pic;
        out = get_reordered_picture(oc, 0);
    }else{
        out = get_reordered_picture(oc, 1);
    }

    if (out){
        if (out->drop){
            oc->dropped_frames++;
        }
        else if (fd){
            do_video_out(oc, fd, out, frame_width, frame_height);
        }else{
#ifdef HAVE_LIBSDL2
            if (h->display){
                display_frame(h, oc, fd, out, frame_width, frame_height, !(pic==NULL));
            }
#endif
        }
        oc->frame_number++;
    }

    return out;
}

OutputContext *get_output_context(H264Context *h){
    const int frame_width=h->frame_width;
    const int frame_height=h->frame_height;
    const int frame_size = frame_width*frame_height;

    OutputContext *oc = av_mallocz(sizeof(OutputContext));
    oc->bit_buffer_size= FFMAX(1024*256, frame_size*2); // oversize a little bit to allow extra border write
    oc->bit_buffer=  av_mallocz(oc->bit_buffer_size);

    return oc;
}

void free_output_context(OutputContext *oc){

    av_free(oc->bit_buffer);
    av_free(oc);
}

SuperMBContext *getSuperMBContext(H264Context *h, int smb_width, int smb_height){
    SuperMBContext *smbc = av_mallocz(sizeof(SuperMBContext));

    smbc->smb_width = smb_width;
    smbc->smb_height = smb_height;

    smbc->nsmb_height = h->mb_height / smbc->smb_height +  (h->mb_height%smbc->smb_height ? 1:0);    //only need one extra if mb_height was not dividable
    smbc->nsmb_width  = h->mb_width / smbc->smb_width;
    while ( (smbc->nsmb_width * smbc->smb_width)-(smbc->smb_height-1) < h->mb_width )
        smbc->nsmb_width++;

    smbc->nsmb_3dheight= smbc->nsmb_height - ((h->mb_height/2)/smbc->smb_height +1); //assuming max motion vector of half the height

    smbc->smbs[0] = av_malloc (smbc->nsmb_width * smbc->nsmb_height * sizeof(SuperMBTask));
    smbc->smbs[1] = av_malloc (smbc->nsmb_width * smbc->nsmb_height * sizeof(SuperMBTask));
    for (int y=0, i=0; i<smbc->nsmb_height; i++, y+=smbc->smb_height){
        for (int x=0, j=0; j<smbc->nsmb_width; j++, x+=smbc->smb_width){
            smbc->smbs[0][i*smbc->nsmb_width +j].smb_y = y;
            smbc->smbs[0][i*smbc->nsmb_width +j].smb_x = x;
            smbc->smbs[1][i*smbc->nsmb_width +j].smb_y = y;
            smbc->smbs[1][i*smbc->nsmb_width +j].smb_x = x;
        }
    }

    smbc->refcount = 1;

    return smbc;
}

void freeSuperMBContext(SuperMBContext *smbc){
    av_free(smbc->smbs[0]);
    av_free(smbc->smbs[1]);
    av_free(smbc);
}

SuperMBContext * acquire_smbc(H264Context *h ){
    SuperMBContext *smbc;

    pthread_mutex_lock (&h->smb_lock);
    smbc = h->smbc;
    smbc->refcount++;
    pthread_mutex_unlock(&h->smb_lock);
    return smbc;
}

void release_smbc(H264Context *h, SuperMBContext *smbc){
    pthread_mutex_lock (&h->smb_lock);
    smbc->refcount--;
    if (smbc->refcount==0){
        freeSuperMBContext(smbc);
    }
    pthread_mutex_unlock(&h->smb_lock);

}


#ifdef HAVE_LIBSDL2

// #if OMPSS
static void draw_sb_border(H264Context *h, uint32_t *rgba_pixels, int smb_x, int smb_y){
    int mb_width = h->mb_width;
    int mb_height = h->mb_height;
    int width = h->frame_width;
    int height = h->frame_height;

    int mb_x = smb_x * h->smb_width;
    int mb_y = smb_y * h->smb_height;

    uint32_t pix= 0x0000FFC0;

    for (int k=0, i=mb_y; i< mb_y + h->smb_height; i++, k++){
        for (int l=0, j=mb_x -k ; j< mb_x - k + h->smb_width; j++, l++){
            //outside frame
            if (i<0 || i>=mb_height || j<0 || j>=mb_width) {
                continue;
            }

            //draw top
            if (i==0 || k==0 || l==0){
                int mx = j*16;
                int my = i*16;
                uint32_t *top = rgba_pixels + my*width + mx;
                int endx = mx+16 < width? 16: width-mx;

                for (int x = 0; x<endx; x++){
                    top[x] = pix;
                }
            }

            //draw bottom
            if (i==mb_height-1 || k==h->smb_height-1 || l==h->smb_width-1){
                int mx = j*16;
                int my = i*16 + 15; my = my < height ? my: height-1;
                uint32_t *bottom = rgba_pixels + my*width + mx;
                int endx = mx+16 < width? 16: width-mx;

                for (int x = 0; x<endx; x++){
                    bottom[x] = pix;
                }
            }

            //draw left
            if (j==0 || l==0 ){
                int mx = j*16;
                int my = i*16;
                uint32_t *left = rgba_pixels + my*width + mx;
                int endy = my +16 < height ? 16: height - my;

                for (int y = 0; y<endy; y++){
                    left[y*width] = pix;
                }
            }

            //draw right
            if (j==mb_width -1 || l==h->smb_width-1 ){
                int mx = j*16 + 15; mx = mx < width ? mx: width-1;
                int my = i*16;
                uint32_t *right = rgba_pixels + my*width + mx;
                int endy = my +16 < height ? 16: height - my;

                for (int y = 0; y<endy; y++){
                    right[y*width] = pix;
                }
            }
        }
    }
}

static void draw_sbmap (H264Context *h, SuperMBContext *smbc, SDLContext *sdlc){
    int pitch;
    uint32_t *rgba_pixels;
    SDL_Texture *sbmap= sdlc->sbmap_texture;

    SDL_LockTexture( sbmap, NULL, (void **)&rgba_pixels, &pitch );

    memset (rgba_pixels, 0, pitch * h->height);
    for (int i=0; i< smbc->nsmb_height; i++){
        for (int j=0; j< smbc->nsmb_width; j++){
            draw_sb_border(h, rgba_pixels, j, i);
        }
    }

    SDL_UnlockTexture( sbmap );
}
// #endif

// static void calc_sb_sizes (H264Context *h, SuperMBContext *smbc){
//     smbc->smb_height = h->smb_height;
//     smbc->smb_width = h->smb_width;
//
//     smbc->nsmb_height = h->mb_height / smbc->smb_height +  (h->mb_height%smbc->smb_height ? 1:0);    //only need one extra if mb_height was not dividable
//     smbc->nsmb_width  = h->mb_width / smbc->smb_width;
//     while ( (smbc->nsmb_width * smbc->smb_width)-(smbc->smb_height-1) < h->mb_width )
//         smbc->nsmb_width++;
// }


static void handle_key_event(H264Context *h, SDLContext *sdlc, SDL_Keysym keysym){
    int arrow=0;

    switch (keysym.sym){
        case SDLK_ESCAPE:
            if (sdlc->fullscreen){
                SDL_SetWindowFullscreen(sdlc->window, SDL_FALSE);
                sdlc->fullscreen = 0;
            }
            break;
        case SDLK_SPACE:
            pthread_mutex_lock(&h->sdl_lock);
            sdlc->pause = !sdlc->pause;
            pthread_cond_signal(&h->sdl_cond);
            pthread_mutex_unlock(&h->sdl_lock);
            break;
        case SDLK_f:
            if (!sdlc->fullscreen){
                if (keysym.mod == KMOD_LCTRL){
//                     SDL_SetWindowDisplayMode (sdlc->window, &sdlc->full);
                    SDL_SetWindowFullscreen(sdlc->window, SDL_TRUE);

                    sdlc->fullscreen = 1;
                }
            }
            break;
        case SDLK_m:
            sdlc->showmap = !sdlc->showmap;
            break;
        case SDLK_UP:
            if (keysym.mod == KMOD_NONE && sdlc->showmap && h->smb_height < h->mb_height && h->smb_height < h->smb_width){
                h->smb_height++;
                arrow =1;
            }
            break;
        case SDLK_DOWN:
            if (keysym.mod == KMOD_NONE && sdlc->showmap && h->smb_height > 1 ){
                h->smb_height--;
                arrow =1;
            }
            break;
        case SDLK_LEFT:
            if (keysym.mod == KMOD_NONE && sdlc->showmap && h->smb_width > 1 && h->smb_width > h->smb_height){
                h->smb_width--;
                arrow =1;
            }
            break;
        case SDLK_RIGHT:
            if (keysym.mod == KMOD_NONE && sdlc->showmap && h->smb_width < h->mb_width){
                h->smb_width++;
                arrow =1;
            }
            break;
    }

    if (arrow){
        SuperMBContext *smbc = getSuperMBContext(h, h->smb_width, h->smb_height);
        pthread_mutex_lock(&h->smb_lock);
        h->smbc->refcount--;
        if (h->smbc->refcount == 0)
            freeSuperMBContext(h->smbc);
        h->smbc = smbc;
        sdlc->updatemap =1;
        pthread_mutex_unlock(&h->smb_lock);
    }
}

void handle_window_event(H264Context *h, SDLContext *sdlc, SDL_WindowEvent winevent){
    SDL_Rect nrect;
    switch (winevent.event){
        case SDL_WINDOWEVENT_RESIZED:

            sdlc->win_w =  winevent.data1;
            sdlc->win_h =  winevent.data2;

            double aspect = (double) sdlc->win_w/ sdlc->win_h;
            if ( aspect < sdlc->aspect){
                double r = (double) sdlc->win_w / sdlc->rect.w;
                double h = (double) sdlc->rect.h * r;

                nrect.y = lrint(( (double) sdlc->win_h - h)/2);
                nrect.h = lrint(h);

                nrect.x=0;
                nrect.w= sdlc->win_w;

            }else {
                double r = (double) sdlc->win_h / sdlc->rect.h;
                double w = (double) sdlc->rect.w * r;

                nrect.x = lrint(( (double) sdlc->win_w - w)/2);
                nrect.w = lrint(w);

                nrect.y=0;
                nrect.h= sdlc->win_h;
            }
            //prob better to lock
            sdlc->win_rect = nrect;
            sdlc->resized=1;
            break;
    }
}

void *sdl_event_listen_thread(void *arg){
    H264Context *h = (H264Context *) arg;
    SDLContext *sdlc = h->sdlc;
    SDL_Event event;

    while ( SDL_WaitEvent(&event) ) {
        switch (event.type) {
            case SDL_KEYDOWN:
                handle_key_event(h, sdlc, event.key.keysym);
                break;
            case SDL_WINDOWEVENT:
                handle_window_event(h, sdlc, event.window);
                break;
            case SDL_QUIT:
                h->quit=1;
                goto finish;
        }
    }
finish:
    pthread_exit(NULL);
    return NULL;
}

//XInitThreads not called in SDL2 library, causes crash
//remove in future when fixed ...
#include <X11/Xlib.h>

SDLContext *get_SDL_context(H264Context *h){
    const int frame_width=h->frame_width;
    const int frame_height=h->frame_height;

    SDLContext *sdlc = av_mallocz(sizeof(SDLContext));
    sdlc->display = h->display;
    sdlc->fullscreen = h->fullscreen;

    sdlc->aspect = (double) frame_width / (double) frame_height;
    sdlc->rect.x =0;
    sdlc->rect.y =0;
    sdlc->rect.w =frame_width;
    sdlc->rect.h =frame_height;

    XInitThreads(); //workaround

    // Initializes the video subsystem
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Unable to init SDL: %s\n", SDL_GetError());
        #undef exit
        exit(-1);
    }
    SDL_SetHint("SDL_HINT_RENDER_SCALE_QUALITY", "best");
    SDL_SetHint("SDL_HINT_RENDER_OPENGL_SHADERS", "1");

    SDL_GetDesktopDisplayMode(0, &sdlc->full);
    sdlc->full.format = SDL_PIXELFORMAT_IYUV;

    sdlc->wind = sdlc->full;
    if (sdlc->wind.w > frame_width) sdlc->wind.w = frame_width;
    if (sdlc->wind.h > frame_height) sdlc->wind.h = frame_height;

    sdlc->win_rect.x =0;
    sdlc->win_rect.y =0;
    sdlc->win_rect.w =sdlc->wind.w;
    sdlc->win_rect.h =sdlc->wind.h;

    if (sdlc->fullscreen){
        sdlc->window = SDL_CreateWindow( h->file_name, SDL_WINDOWPOS_UNDEFINED,  SDL_WINDOWPOS_UNDEFINED, sdlc->full.w, sdlc->full.h, SDL_WINDOW_FULLSCREEN|SDL_WINDOW_SHOWN|SDL_WINDOW_RESIZABLE);
        SDL_SetWindowDisplayMode (sdlc->window, &sdlc->full);
    } else {
        sdlc->window = SDL_CreateWindow( h->file_name, SDL_WINDOWPOS_UNDEFINED,  SDL_WINDOWPOS_UNDEFINED, sdlc->wind.w, sdlc->wind.h, SDL_WINDOW_RESIZABLE|SDL_WINDOW_SHOWN);
        SDL_SetWindowDisplayMode (sdlc->window, &sdlc->wind);
    }

    sdlc->renderer = SDL_CreateRenderer(sdlc->window, -1, SDL_RENDERER_ACCELERATED);
//     sdlc->renderer = SDL_CreateRenderer(sdlc->window, -1, SDL_RENDERER_SOFTWARE);

    h->sdlq.queue[0] = SDL_CreateTexture (sdlc->renderer, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING, frame_width, frame_height);
    h->sdlq.queue[1] = SDL_CreateTexture (sdlc->renderer, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING, frame_width, frame_height);

    sdlc->sbmap_texture = SDL_CreateTexture (sdlc->renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, frame_width, frame_height);
    SDL_SetTextureBlendMode(sdlc->sbmap_texture, SDL_BLENDMODE_BLEND);
    sdlc->updatemap = 1;

#if HAVE_LIBSDL_TTF
    //not working with SDL 2.0, try again in future when supported
    if(TTF_Init()==-1) {
        printf("TTF_Init: %s\n", TTF_GetError());
        exit(2);
    }

    // Load a font
    TTF_Font *font;
    font = TTF_OpenFont("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 24);
    if (font == NULL)
    {
        printf("TTF_OpenFont() Failed: %s\n", TTF_GetError());
        TTF_Quit();
        exit(1);
    }
#endif

    pthread_create(&sdlc->listen_thread, NULL, sdl_event_listen_thread, h);

    return sdlc;

}

void free_SDL_context(H264Context *h){
    SDLContext *sdlc = h->sdlc;
    pthread_join(sdlc->listen_thread, NULL);

#if HAVE_LIBSDL_TTF
    TTF_Quit();
#endif
    SDL_DestroyTexture(h->sdlq.queue[0]);
    SDL_DestroyTexture(h->sdlq.queue[1]);
    SDL_DestroyTexture(sdlc->sbmap_texture);
    SDL_DestroyRenderer(sdlc->renderer);
    SDL_DestroyWindow(sdlc->window);
    SDL_Quit();

}

void *sdl_thread(void *arg){
    H264Context *h = (H264Context *) arg;

    SDLContext *sdlc = get_SDL_context(h);
    h->sdlc = sdlc;

    signal_texture(h, 0);
    signal_texture(h, 0);

    SDL_Texture *texture;
    for (;;){
        pthread_mutex_lock(&h->sdl_lock);
        while (sdlc->pause){
            pthread_cond_wait(&h->sdl_cond, &h->sdl_lock);
        }
        pthread_mutex_unlock(&h->sdl_lock);

        texture = get_next_texture(h, 0);
        if (texture == NULL)
            break;

        SDL_UnlockTexture(texture);

        //clear if resized
        if (sdlc->resized){
            // KDE bug prob, reset viewport change after resize from max
            SDL_RenderSetViewport(sdlc->renderer, NULL);
            SDL_SetRenderDrawColor(sdlc->renderer, 0, 0, 0, 255);
            SDL_RenderClear(sdlc->renderer);
            sdlc->resized = 0;
        }

        SDL_RenderCopy(sdlc->renderer, texture, &sdlc->rect, &sdlc->win_rect);

        if (sdlc->showmap){
            if (sdlc->updatemap){
                SuperMBContext *smbc;
                pthread_mutex_lock (&h->smb_lock);
                smbc = h->smbc;
                smbc->refcount++;
                sdlc->updatemap=0;
                pthread_mutex_unlock(&h->smb_lock);

                draw_sbmap(h, smbc, sdlc);

                release_smbc(h, smbc);
            }
            SDL_RenderCopy(sdlc->renderer, sdlc->sbmap_texture, &sdlc->rect, &sdlc->win_rect);
        }

        SDL_RenderPresent(sdlc->renderer);
        signal_texture(h, 0);
    }

    free_SDL_context(h);

    pthread_exit(NULL);
    return NULL;
}
#endif

