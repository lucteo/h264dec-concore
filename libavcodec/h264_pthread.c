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
#include "h264_parser.h"
#include "h264_nal.h"
#include "h264_entropy.h"
#include "h264_rec.h"
#include "h264_misc.h"
// #undef NDEBUG
#include <assert.h>
#include <pthread.h>

#define XOANON 0

#ifdef XOANON
static int ed_rec_affinity[40] = { 0,  4,  8, 12, 16, 20, 24, 28, 32, 36,
                                   1,  5,  9, 13, 17, 21, 25, 29, 33, 37,
                                   2,  6, 10, 14, 18, 22, 26, 30, 34, 38,
                                   3,  7, 11, 15, 19, 23, 27, 31, 35, 39 };
static int ed_rec_smt_aff[80]  = { 0,  40,  4, 44,  8, 48, 12, 52, 16, 56, 20, 60, 24, 64, 28, 68, 32, 72, 36, 76,
                                   1,  41,  5, 45,  9, 49, 13, 53, 17, 57, 21, 61, 25, 65, 29, 69, 33, 73, 37, 77,
                                   2,  42,  6, 46, 10, 50, 14, 54, 18, 58, 22, 62, 26, 66, 30, 70, 34, 74, 38, 78,
                                   3,  43,  7, 47, 11, 51, 15, 55, 19, 59, 23, 63, 27, 67, 31, 71, 35, 75, 39, 79 };
#else
static int ed_rec_affinity[12] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11};
static int ed_rec_smt_aff[12] = { 0,  6,  1,  7,  2,  8,  3,  9,  4,  10, 5, 11};
#endif

static int frames=0;

static void notify_one_worker(H264Context *h){
    pthread_mutex_lock(&h->task_lock);
    pthread_cond_signal(&h->task_cond);
    pthread_mutex_unlock(&h->task_lock);
}

static void notify_all_workers(H264Context *h){
    pthread_mutex_lock(&h->task_lock);
    pthread_cond_broadcast(&h->task_cond);
    pthread_mutex_unlock(&h->task_lock);
}

static void push_sbe (SliceBufferQueue *sbq, SliceBufferEntry *sbe, int notify ){
    pthread_mutex_lock(&sbq->lock);
    while (sbq->cnt >= sbq->size)
        pthread_cond_wait(&sbq->cond, &sbq->lock);
    sbq->queue[sbq->fi] = sbe;
    sbq->cnt++;
    sbq->fi++; sbq->fi %= sbq->size;
    if (notify)
        pthread_cond_signal(&sbq->cond);
    pthread_mutex_unlock(&sbq->lock);
}

static SliceBufferEntry* pop_sbe (SliceBufferQueue *sbq, int block){
    SliceBufferEntry *sbe=NULL;

    pthread_mutex_lock(&sbq->lock);
    if (block){
        while (sbq->cnt <= 0)
            pthread_cond_wait(&sbq->cond, &sbq->lock);
    }else {
        if (sbq->cnt <= 0)
            goto nonblock;
    }
    sbe = sbq->queue[sbq->fo];
    sbq->cnt--;
    sbq->fo++; sbq->fo %= sbq->size;
    pthread_cond_signal(&sbq->cond);
nonblock:
    pthread_mutex_unlock(&sbq->lock);

    return sbe;
}

// static void push_rle (RingLineQueue *rlq, SliceBufferEntry *sbe, int line, int notify){
//
//     //check for free slots
//     pthread_mutex_lock(&rlq->wslock);
//     while (rlq->free <= 0){
//         pthread_cond_wait(&rlq->wscond, &rlq->wslock);
//     }
//     //free slot is available, decrement one in this lock
//     rlq->free--;
//     pthread_mutex_unlock(&rlq->wslock);
//
//     pthread_mutex_lock(&rlq->swlock);
//     rlq->queue[rlq->fi]->sbe=sbe;
//     rlq->queue[rlq->fi]->line=line;
//     rlq->queue[rlq->fi]->mb_cnt=0;
//     rlq->fi++; rlq->fi %= rlq->size;
//     rlq->ready++;
//     if(notify)
//         pthread_cond_signal(&rlq->swcond);
//     pthread_mutex_unlock(&rlq->swlock);
// }

// static RingLineEntry* pop_rle (RingLineQueue *rlq, int block){
//     RingLineEntry *rle=NULL;
//
//     pthread_mutex_lock(&rlq->swlock);
//     if (block){
//         while (rlq->ready <= 0)
//             pthread_cond_wait(&rlq->swcond, &rlq->swlock);
//     }else {
//         if (rlq->ready <= 0)
//             goto nonblock;
//     }
//     rle = rlq->queue[rlq->fo];
//     rlq->fo++; rlq->fo %= rlq->size;
//     rlq->ready--;
// nonblock:
//     pthread_mutex_unlock(&rlq->swlock);
//
//     return rle;
// }
//
// static void rel_rle (RingLineQueue *rlq){
//     pthread_mutex_lock(&rlq->wslock);
//     rlq->free++;
//     pthread_cond_signal(&rlq->wscond);
//     pthread_mutex_unlock(&rlq->wslock);
// }

static RingLineEntry* pop_rle (SliceBufferQueue *sbq, RingLineQueue *rlq, int *has_token){
    RingLineEntry *rle=NULL;
    SliceBufferEntry *sbe=NULL;
    int line=-1;

    pthread_mutex_lock(&sbq->lock);
    if (sbq->cnt <= 0)
        goto unlock;
    sbe = sbq->queue[sbq->fo];
    line = sbe->lines_taken;


    pthread_mutex_lock(&rlq->swlock);
    if (!*has_token){
        if (rlq->free <= 0)
            goto unlock2;
        rlq->free--;
        *has_token=1;
    }
    rle = rlq->queue[rlq->fo];
    rlq->fo++; rlq->fo %= rlq->size;
    rle->sbe=sbe;
    rle->line = line;
    rle->mb_cnt =0;
    if (++sbe->lines_taken >= sbe->lines_total){
        sbq->cnt--;
        sbq->fo++; sbq->fo %= sbq->size;
        pthread_cond_signal(&sbq->cond);
    }
unlock2:
    pthread_mutex_unlock(&rlq->swlock);
unlock:
    pthread_mutex_unlock(&sbq->lock);


    return rle;
}

static void rel_rle (RingLineQueue *rlq, int *rec_token){
    pthread_mutex_lock(&rlq->swlock);
    rlq->free++;
    *rec_token=0;
//     pthread_cond_signal(&rlq->swcond);
    pthread_mutex_unlock(&rlq->swlock);

}

//get either a entropy or a line reconstruct task
static void pop_next_task(H264Context *h, SliceBufferEntry **psbe, RingLineEntry **prle, int *rec_token){

    pthread_mutex_lock(&h->task_lock);

    for(;;){
        if ( (*psbe = pop_sbe(&h->sb_q[ENTROPY], 0)) ){
            if (*rec_token){
                rel_rle(&h->rl_q, rec_token);
                pthread_cond_signal(&h->task_cond);
            }
            break;
        }
        else if ( (*prle = pop_rle(&h->sb_q[MBDEC], &h->rl_q, rec_token)) )
            break;
        pthread_cond_wait(&h->task_cond, &h->task_lock);
    }

    pthread_mutex_unlock(&h->task_lock);
}

void *parse_thread(void *arg){
    H264Context *h = (H264Context *) arg;
    ParserContext *pc = get_parse_context(h->ifile);
    NalContext *nc = get_nal_context(h->width, h->height);
    H264Slice *s;
    SliceBufferEntry *sbe = NULL;

    while(!pc->final_frame && frames++ <h->num_frames && !h->quit){
        sbe = get_sb_entry(h);

        av_read_frame_internal(pc, &sbe->gb);
        s = &sbe->slice;

        decode_nal_units(nc, s, &sbe->gb);

        push_sbe(&h->sb_q[ENTROPY], sbe, 0);
        notify_one_worker(h);
    }

    if (!h->no_mbd){
        sbe = get_sb_entry(h);
        sbe->state=-1;
        sbe->slice.coded_pic_num=nc->coded_pic_num;
        sbe->lines_total=h->threads;

        push_sbe(&h->sb_q[REORDER], sbe, 1);
    }else{
        for (int i=0; i<h->threads; i++){
            sbe = get_sb_entry(h);
            sbe->state=-1;
            push_sbe(&h->sb_q[ENTROPY], sbe, 1);
            notify_one_worker(h);
        }
    }
    free_nal_context(nc);
    free_parse_context(pc);

    pthread_exit(NULL);
    return NULL;
}

int decode_slice_entropy(EntropyContext *ec, SliceBufferEntry *sbe){
    int i,j;
    H264Slice *s = &sbe->slice;
    GetBitContext *gb = &sbe->gb;
    CABACContext *c = &ec->c;
    H264Mb *mbs = sbe->mbs;

    if( !s->pps.cabac ){
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return -1;
    }

    init_dequant_tables(s, ec);
    ec->curr_qscale = s->qscale;
    ec->last_qscale_diff = 0;
    ec->chroma_qp[0] = get_chroma_qp( s, 0, s->qscale);
    ec->chroma_qp[1] = get_chroma_qp( s, 1, s->qscale);

    /* realign */
    align_get_bits( gb );
    /* init cabac */
    ff_init_cabac_decoder( c, gb->buffer + get_bits_count(gb)/8, (get_bits_left(gb) + 7)/8);

    ff_h264_init_cabac_states(ec, s, c);

    for(j=0; j<ec->mb_height; j++){
        init_entropy_buf(ec, s, j);
        for(i=0; i<ec->mb_width; i++){
            int eos,ret;
            H264Mb *m = &mbs[i + j*ec->mb_width];
            //memset(m, 0, sizeof(H264Mb));
            m->mb_x=i;
            m->mb_y=j;
            ec->m = m;

            ret = ff_h264_decode_mb_cabac(ec, s, c);
            eos = get_cabac_terminate( c); (void) eos;

            if( ret < 0 || c->bytestream > c->bytestream_end + 2) {
                av_log(AV_LOG_ERROR, "error while decoding MB %d %d, bytestream (%td)\n", m->mb_x, m->mb_y, c->bytestream_end - c->bytestream);
                return -1;
            }
        }
    }

    return 0;
}

static int decode_slice_mb(MBRecContext *d, RingLineEntry *rle, int frames){
    SliceBufferEntry *sbe= rle->sbe;
    H264Slice *s = &sbe->slice;
    H264Mb *mbs = sbe->mbs;

    int mb_width= d->mb_width;
    int i;
    const int line = rle->line;

    init_mbrec_context(d, d->mrs, s, line);

    H264Mb *m = &mbs[line*mb_width];
    d->top=rle->prev_line->top;
    d->top_next=rle->top;

//     assert(rle->mb_cnt ==0);
    for(i=0; i< mb_width; i++){
        if (frames || line>0){
            while (rle->mb_cnt >= rle->prev_line->mb_cnt -1);
        }
        h264_decode_mb_internal( d, d->mrs, s, &m[i]);
        rle->mb_cnt++;
    }
    draw_edges(d, s, line);

    return 0;
}

// static int decode_slice_mb_static(MBRecContext *d, H264Slice *s, RLThreadContext *r, RLThreadContext *rp,  int frames){
//     int mb_height= d->mb_height;
//     int mb_width= d->mb_width;
//     int thread_num = r->thread_num;
//     int thread_total = r->thread_total;
//     int i;
//     int j = thread_num;
//
//     r->mb_cnt=frames* mb_height*mb_width;
//     for(; j<mb_height; j+=thread_total){
//         H264Mb *m = &s->mbs[j*mb_width];
//         for(i=0; i< mb_width; i++){
//             if (j>0){
//                 while (r->mb_cnt- (thread_num? 0:mb_width) >= rp->mb_cnt-1);
//             }
//             h264_decode_mb_internal(d, s, m++);
//             r->mb_cnt++;
//         }
//         draw_edges(d, s, j);
//     }
//     return 0;
// }

static void *ed_rec_thread(void *arg){
    H264Context *h =  (H264Context*) arg;
    EntropyContext *ec=NULL;
    MBRecContext *mrc=NULL;

    RingLineEntry *rle=NULL;
    SliceBufferEntry *sbe=NULL;
    H264Slice *s;
    int rec_token=0;

    if (!h->no_mbd){
        mrc = get_mbrec_context(h);
    }
    ec = get_entropy_context(h);

    for(;;){
        pop_next_task(h, &sbe, &rle, &rec_token);
        if (sbe){
            if (h->no_mbd && sbe->state<0){
                break;
            }
            if (!sbe->initialized){
                init_sb_entry(h, sbe);
            }
            decode_slice_entropy(ec, sbe);

            if (h->no_mbd){
                release_sb_entry(h, sbe);
                sbe=NULL;
            } else {
                push_sbe(&h->sb_q[REORDER], sbe, 1);
            }
        } else if (rle){
            if (rle->sbe->state<0)
                break;
            s = &rle->sbe->slice;

            decode_slice_mb(mrc, rle, s->coded_pic_num);

            if (rle->line == h->mb_height-1){
                push_sbe(&h->sb_q[OUTPUT], rle->sbe, 1);
            }
            rle->mb_cnt++;
        }
    }

    //make sure threads quit in order of rle assignment
    if (!h->no_mbd){
        while (rle->prev_line->mb_cnt <= h->mb_width);
        rel_rle(&h->rl_q, &rec_token);
        notify_one_worker(h);
        rle->mb_cnt = h->mb_width +1;
        if (rle->line == h->threads-1){
            push_sbe(&h->sb_q[OUTPUT], rle->sbe, 1);
        }

        free_mbrec_context(mrc);
    }

    free_entropy_context(ec);

    pthread_exit(NULL);
    return NULL;
}

static void *reorder_thread(void *arg){
    H264Context *h = (H264Context *) arg;
    int i;
    SliceBufferEntry *reorder[h->sb_size];
    SliceBufferEntry *sbe, *next_sbe;
    H264Slice *s;
    int reorder_cnt=0;
    unsigned next_pic_num=0;

    for(;;){

        sbe = pop_sbe(&h->sb_q[REORDER], 1);

        s = &sbe->slice;
        for(i=reorder_cnt; i>0; i--){
            if (s->coded_pic_num < reorder[i-1]->slice.coded_pic_num)
                break;
            reorder[i]=reorder[i-1];
        }
        reorder[i]=sbe;

        while(reorder_cnt>=0){
            if (next_pic_num!=reorder[reorder_cnt]->slice.coded_pic_num){
                break;
            }
            next_sbe = reorder[reorder_cnt];
            H264Slice *es = &next_sbe->slice;

            if (next_sbe->state<0)
                goto end;

            for (int i=0; i<2; i++){
                for(int j=0; j< es->ref_count[i]; j++){
                    if (es->ref_list_cpn[i][j] ==-1)
                        continue;
                    int k;
                    for (k=0; k<h->max_dpb_cnt; k++){
                        if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == es->ref_list_cpn[i][j]){
                            es->dp_ref_list[i][j] = &h->dpb[k];
                            break;
                        }
                    }
                }
            }
            next_sbe->dp = get_dpb_entry(h, es);

            push_sbe(&h->sb_q[MBDEC], next_sbe, 0);
            notify_all_workers(h);

//             for (int i=0; i< h->mb_height; i++){
//                 push_rle(&h->rl_q, next_sbe, i, 0);
//                 notify_one_worker(h);
//             }


            next_pic_num++;
            reorder_cnt--;
        }
        reorder_cnt++;
    }

end:
    {
        push_sbe(&h->sb_q[MBDEC], next_sbe, 0);
        notify_all_workers(h);
        if (h->no_mbd){
            push_sbe(&h->sb_q[OUTPUT], next_sbe, 1);
        }
//         for (int i=0; i< h->threads; i++){
//             push_rle(&h->rl_q, next_sbe, i, 0);
//             notify_one_worker(h);
//         }
    }

    pthread_exit(NULL);
    return NULL;
}

void create_ed_rec_threads(H264Context *h){
    cpu_set_t cpuset;
    int* aff;

    if (h->setaff ==1){
        for (int i=0; i<h->threads; i++){
            pthread_attr_init(&h->ed_rec_attr[i]);
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            pthread_attr_setaffinity_np(&h->ed_rec_attr[i], sizeof(cpu_set_t), &cpuset);
            pthread_create(&h->ed_rec_thr[i], &h->ed_rec_attr[i], ed_rec_thread, h);
        }
    } else if (h->setaff ==2){
        aff = h->smt ? ed_rec_smt_aff : ed_rec_affinity ;
        for (int i=0; i<h->threads; i++){
            pthread_attr_init(&h->ed_rec_attr[i]);
            CPU_ZERO(&cpuset);
            CPU_SET(aff[i], &cpuset);
            pthread_attr_setaffinity_np(&h->ed_rec_attr[i], sizeof(cpu_set_t), &cpuset);
            pthread_create(&h->ed_rec_thr[i], &h->ed_rec_attr[i], ed_rec_thread, h);
        }
    } else {
        for (int i=0; i<h->threads; i++){
            pthread_create(&h->ed_rec_thr[i], NULL, ed_rec_thread, h);
        }
    }
}

void join_ed_rec_threads(H264Context *h){
    for (int i=0; i< h->threads; i++){
        pthread_join(h->ed_rec_thr[i], NULL);
    }
}

void *output_thread(void *arg){
    H264Context *h = (H264Context *) arg;

    OutputContext *oc = get_output_context( h );

    SliceBufferEntry *sbe = NULL;
    H264Slice *s=NULL;
    for(;;) {
        DecodedPicture *out, *dp;
        sbe = pop_sbe(&h->sb_q[OUTPUT], 1);

        if (sbe->state <0)
            break;

        s = &sbe->slice;
        for (int i=0; i<s->release_cnt; i++){
            for(int j=0; j<h->max_dpb_cnt; j++){
                if(h->dpb[j].cpn== s->release_ref_cpn[i]){
                    release_dpb_entry(h, &h->dpb[j], 2);
                    break;
                }
            }
        }

        dp=sbe->dp;
        release_sb_entry(h, sbe);

        out =output_frame(h, oc, dp, h->ofile, h->frame_width, h->frame_height);
        if (out){
            release_dpb_entry(h, out, 1);
        }

        print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 0, h->verbose);

    }
    /* at the end of stream, we must flush the decoder buffers */
    while (output_frame(h, oc, NULL, h->ofile, h->frame_width, h->frame_height));
    print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 1, h->verbose);

    free_output_context(oc);

    pthread_exit(NULL);
    return NULL;
}

/*
* The following code is the main loop of the file converter
*/
int h264_decode_pthread(H264Context *h) {
    pthread_t parse_thr, reorder_thr, output_thr;

    av_start_timer();

    pthread_create(&parse_thr, NULL, parse_thread, h);
    if (!h->no_mbd){
        pthread_create(&reorder_thr, NULL, reorder_thread, h);
        pthread_create(&output_thr, NULL, output_thread, h);
    }
#if HAVE_LIBSDL2
    pthread_t sdl_thr;
    if (h->display){
        pthread_create(&sdl_thr, NULL, sdl_thread, h);
    }
#endif
    create_ed_rec_threads(h);


    if (h->rl_side_touch){
        pthread_mutex_lock(&h->ilock);
        while (h->init_threads< h->threads)
            pthread_cond_wait(&h->icond, &h->ilock);
        pthread_mutex_unlock(&h->ilock);

        pthread_mutex_lock(&h->tlock);
        h->touch_start =1;
        pthread_cond_broadcast(&h->tcond);
        pthread_mutex_unlock(&h->tlock);

        pthread_mutex_lock(&h->tdlock);
        while (h->touch_done < h->threads)
            pthread_cond_wait(&h->tdcond, &h->tdlock);
        pthread_mutex_unlock(&h->tdlock);

        pthread_mutex_lock(&h->slock);
        h->start =1;
        pthread_cond_broadcast(&h->scond);
        pthread_mutex_unlock(&h->slock);
    }
    join_ed_rec_threads(h);
    pthread_join(parse_thr, NULL);
    if (!h->no_mbd){
        pthread_join(reorder_thr, NULL);
        pthread_join(output_thr, NULL);
    }
#if HAVE_LIBSDL2
    if (h->display)
        signal_sdl_exit(h);
        pthread_join(sdl_thr, NULL);
#endif


    return 0;
}
