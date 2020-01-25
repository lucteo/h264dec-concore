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

#include "h264_types.h"
#include "h264_parser.h"
#include "h264_nal.h"
#include "h264_entropy.h"
#include "h264_rec.h"
#include "h264_pred_mode.h"
#include "h264_misc.h"
#include <assert.h>
#include <time.h>

static void parse_task(H264Context *h, ParserContext *pc, NalContext *nc, SliceBufferEntry *sbe){
    H264Slice *s;

    if (!sbe->initialized){
        init_sb_entry(h, sbe);
        sbe->lines_total=h->mb_height;
    }

    av_read_frame_internal(pc, &sbe->gb);
    s = &sbe->slice;

    decode_nal_units(nc, s, &sbe->gb);

    sbe->drop =0;
    sbe->droppable =0;
    if (h->controlfps){
        if (!s->nal_ref_idc){
            sbe->droppable=1;
            sbe->deadline = h->nexttime + h->framedelay;
        }
    }
}

static void decode_slice_entropy_task(H264Context *h, EntropyContext *ec, SliceBufferEntry *sbe){
    int i,j;
    H264Slice *s = &sbe->slice;
    GetBitContext *gb = &sbe->gb;
    H264Mb *mbs = sbe->mbs;
//     GetBitContext *gb = s->gb;
    CABACContext *c = &ec->c;

    if( !s->pps.cabac ){
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return ;
    }

    if (sbe->droppable) {
        struct timespec curtime;
        int64_t curtime_ns;
        clock_gettime(CLOCK_REALTIME, &curtime);
        curtime_ns = curtime.tv_sec * 1000000000 + curtime.tv_nsec;
        if (curtime_ns > sbe->deadline  ){
            sbe->drop = 1; //optimize checking for following tasks
            return;
        }
    }

    init_dequant_tables(s, ec);
    ec->curr_qscale = s->qscale;
    ec->last_qscale_diff = 0;
    ec->chroma_qp[0] = get_chroma_qp((H264Slice *) s, 0, s->qscale);
    ec->chroma_qp[1] = get_chroma_qp((H264Slice *) s, 1, s->qscale);

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
            m->mb_x=i;
            m->mb_y=j;
            ec->m = m;

            ret = ff_h264_decode_mb_cabac(ec, s, c);
            eos = get_cabac_terminate( c);
            (void) eos;
            if( ret < 0 || c->bytestream > c->bytestream_end + 2) {
                av_log(AV_LOG_ERROR, "error while decoding MB %d %d, bytestream (%td)\n", m->mb_x, m->mb_y, c->bytestream_end - c->bytestream);
                return ;
            }
        }
    }
}

static void decode_super_mb_block(MBRecContext *d, SliceBufferEntry* sbe, SuperMBContext *smbc, int smb_x, int smb_y){
    MBRecState mrs;
//     memset(&mrs, 0, sizeof(MBRecState));
    H264Slice *s = &sbe->slice;
    H264Mb *mbs = sbe->mbs;

    if (sbe->drop) return;
    if (sbe->droppable) {
        struct timespec curtime;
        int64_t curtime_ns;
        clock_gettime(CLOCK_REALTIME, &curtime);
        curtime_ns = curtime.tv_sec * 1000000000 + curtime.tv_nsec;
        if ( curtime_ns > sbe->deadline ){
            sbe->drop = 1; //optimize checking for following tasks
            return;
        }
    }

    for (int k=0, i= smb_y; i< smb_y + smbc->smb_height; i++, k++){
        init_mbrec_context(d, &mrs, s, i);
        for (int j= smb_x -k ; j< smb_x - k + smbc->smb_width; j++){
            if (i< d->mb_height && j >= 0 && j < d->mb_width){
                h264_decode_mb_internal (d, &mrs, s, &mbs[i*d->mb_width+j]);
            }
        }
    }
}

#pragma omp task input(*d, *sbe, *ml, *mur) inout(*m)
static void decode_super_mb_task(MBRecContext *d, SliceBufferEntry *sbe, SuperMBContext *smbc, SuperMBTask *ml,
SuperMBTask *mur, SuperMBTask *m){
    decode_super_mb_block(d, sbe, smbc, m->smb_x, m->smb_y);
}

#pragma omp task input(*d, *sbe) inout(*sm)
static void draw_edges_task(MBRecContext *d, SliceBufferEntry *sbe, SuperMBContext *smbc, SuperMBTask *sm, int line){
    H264Slice *s = &sbe->slice;
    for (int i=line*smbc->smb_height; i< (line+1)*smbc->smb_height && i< d->mb_height; i++)
        draw_edges(d, s, i);
}

static void decode_mb_in_slice(H264Context *h, MBRecContext *d, SliceBufferEntry *sbe){
    int i,j;

    SuperMBContext *smbc = acquire_smbc(h);
    int smb_height =smbc->nsmb_height, smb_width= smbc->nsmb_width;
    SuperMBTask *smbs = smbc->smbs[0];

    SuperMBTask *sm=NULL, *sml, *smur;
    for(j=0; j< smb_height; j++){
        for(i=0; i< smb_width; i++){
            sm = smbs + j*smb_width + i;
            sml  = sm - ((i > 0) ? 1: 0);
            smur = sm + (((i < smb_width-1) && (j >0))  ? -smb_width+1: 0);
            decode_super_mb_task(d, sbe, smbc, sml, smur, sm);
        }
        draw_edges_task(d, sbe, smbc, sm, j);
    }
    #pragma omp taskwait on(*sm)

    release_smbc(h, smbc);
}

static void decode_slice_mb_task(H264Context *h, MBRecContext *d, SliceBufferEntry *sbe){
    H264Slice *s = &sbe->slice;

    for (int i=0; i<2; i++){
        for(int j=0; j< s->ref_count[i]; j++){
            if (s->ref_list_cpn[i][j] ==-1)
                continue;
            int k;
            for (k=0; k< h->max_dpb_cnt; k++){
                if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == s->ref_list_cpn[i][j]){
                    s->dp_ref_list[i][j] = &h->dpb[k];
                    break;
                }
            }
        }
    }

    #pragma omp critical (dpb)
    s->curr_pic = get_dpb_entry(h, s);

    if (!h->no_mbd){
        decode_mb_in_slice (h, d, sbe);
    }

    for (int i=0; i<s->release_cnt; i++){
        for(int j=0; j<h->max_dpb_cnt; j++){
            if(h->dpb[j].cpn== s->release_ref_cpn[i]){
                #pragma omp critical (dpb)
                release_dpb_entry(h, &h->dpb[j], 2);
                break;
            }
        }
    }
    s->release_cnt=0;
}

// for static 3d wave
/*-------------------------------------------------------------------------------*/
#pragma omp task input(*d, *sbe, *ml, *mur, *mprev) inout(*m)
static void decode_3dwave_super_mb_task(MBRecContext *d, SliceBufferEntry *sbe, SuperMBContext *smbc, SuperMBTask *ml,
SuperMBTask *mur, SuperMBTask *mprev, SuperMBTask *m){
    decode_super_mb_block(d, sbe, smbc, m->smb_x, m->smb_y);
}

// int init_ref_count=0;
#pragma omp task inout(*d, *sbe, *init)
static void init_ref_list_and_get_dpb_task(H264Context *h, MBRecContext *d, SliceBufferEntry *sbe, int *init){
    H264Slice *s = &sbe->slice;
    for (int i=0; i<2; i++){
        for(int j=0; j< s->ref_count[i]; j++){
            if (s->ref_list_cpn[i][j] ==-1)
                continue;
            int k;
            for (k=0; k<h->max_dpb_cnt; k++){
                if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == s->ref_list_cpn[i][j]){
                    s->dp_ref_list[i][j] = &h->dpb[k];
                    break;
                }
            }
        }
    }

    #pragma omp critical (dpb)
    get_dpb_entry(h, s);

}

static SuperMBTask* add_decode_slice_3dwave_tasks(MBRecContext *d, SliceBufferEntry *sbe, SuperMBContext *smbc){
    int i,j;

    int smb_3d_height =smbc->nsmb_3dheight;
    int smb_height =smbc->nsmb_height, smb_width= smbc->nsmb_width;
    int smb_diff_prev = smb_height - smb_3d_height;
    SuperMBTask *sm=NULL, *sml, *smur, *smprev;

    SuperMBTask *smbs = smbc->smbs[smbc->index++]; smbc->index%=2;
    SuperMBTask *smbs_prev = smbc->smbs[smbc->index]; // index rotates -> next == prev

    for(j=0; j<smb_3d_height ; j++){
        for(i=0; i< smb_width; i++){
            sm = smbs + j*smb_width + i;
            sml  = sm - ((i > 0) ? 1: 0);
            smur = sm + (((i < smb_width-1) && (j >0))  ? -smb_width+1: 0);
            smprev = smbs_prev + (j + smb_diff_prev+1)*smb_width -1;
            decode_3dwave_super_mb_task(d, sbe, smbc, sml, smur, smprev, sm);
        }
        draw_edges_task(d, sbe, smbc, sm, j);
    }

    for(; j< smb_height; j++){
        for(i=0; i< smb_width; i++){
            sm = smbs + j*smb_width + i;
            sml  = sm - ((i > 0) ? 1: 0);
            smur = sm + (((i < smb_width-1) && (j >0))  ? -smb_width+1: 0);
            decode_super_mb_task(d, sbe, smbc, sml, smur, sm);
        }
        draw_edges_task(d, sbe, smbc, sm, j);
    }
    return sm;
}

#pragma omp task inout(*d, *sbe, *release) input (*lastsmb)
static void release_ref_list_task(H264Context *h, SuperMBContext *smbc, MBRecContext *d, SliceBufferEntry *sbe, SuperMBTask *lastsmb, int *release){
    H264Slice *s = &sbe->slice;
    for (int i=0; i<s->release_cnt; i++){
        for(int j=0; j<h->max_dpb_cnt; j++){
            if(h->dpb[j].cpn== s->release_ref_cpn[i]){
                #pragma omp critical (dpb)
                release_dpb_entry(h, &h->dpb[j], 2);
                break;
            }
        }
    }
    s->release_cnt=0;

    release_smbc(h, smbc);

}

// static void decode_mb_static_3dwave(H264Context *h, int mb_height, int mb_width, MBRecContext *d, H264Slice *s, H264Mb *mbs, SuperMBTask *smbs, SuperMBTask *smbs_prev){
//
// }
/*-------------------------------------------------------------------------------*/
//end for static 3d wave

static void output_task(H264Context *h, OutputContext *oc, SliceBufferEntry *sbe){
    if (sbe->drop)
        sbe->slice.curr_pic->drop=1;
    DecodedPicture* out =output_frame(h, oc, sbe->slice.curr_pic, h->ofile, h->frame_width, h->frame_height);
    if (out){
        #pragma omp critical (dpb)
        release_dpb_entry(h, out, 1);
    }
    print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 0, h->verbose);
}

/*
* The following code is the main loop of the file converter
*/
int h264_decode_ompss( H264Context *h) {
    const int bufs = h->pipe_bufs;

    ParserContext *pc;
    NalContext *nc;
    EntropyContext *ec[bufs];
    MBRecContext *rc[2];
    OutputContext *oc;
    SliceBufferEntry *sbe;
    SuperMBContext *smbc;

    DecodedPicture *out;
    int frames=0;

#if HAVE_LIBSDL2
    pthread_t sdl_thr;
    if (h->display){
        pthread_create(&sdl_thr, NULL, sdl_thread, h);
    }
#endif
    sbe= av_mallocz(sizeof(SliceBufferEntry) * bufs);


    pc = get_parse_context(h->ifile);
    nc = get_nal_context(h->width, h->height);

    for(int i=0; i<bufs; i++){
        ec[i] = get_entropy_context( h );
    }

    for(int i=0; i<2; i++){
        rc[i] = get_mbrec_context(h);
    }

    oc = get_output_context( h );

    av_start_timer();
    int k=0; int init, release;
    if (h->static_3d && bufs < h->num_frames ){
        int num_pre_ed =0;
        for (num_pre_ed=0; num_pre_ed< bufs -1 && !pc->final_frame; num_pre_ed++){
            #pragma omp taskwait on(sbe[k%bufs])
            parse_task( h, pc, nc, &sbe[k%bufs] ); //not task anymore
            #pragma omp task inout(*ec[k%bufs]) inout(sbe[k%bufs])
            decode_slice_entropy_task(h, ec[k%bufs], &sbe[k%bufs]);            
            k++;
            h->nexttime = h->nexttime + h->frametime;
        }

        while(!pc->final_frame && frames++ < h->num_frames && !h->quit){
            #pragma omp taskwait on(sbe[k%bufs])
            parse_task( h, pc, nc, &sbe[k%bufs] ); //not task anymore
            #pragma omp task inout(*ec[k%bufs]) inout(sbe[k%bufs])
            decode_slice_entropy_task(h, ec[k%bufs], &sbe[k%bufs]);

            k++;

            init_ref_list_and_get_dpb_task(h, rc[k%2], &sbe[k%bufs], &init);
            smbc = acquire_smbc(h);
            SuperMBTask *lastsmb= add_decode_slice_3dwave_tasks(rc[k%2], &sbe[k%bufs], smbc);
            release_ref_list_task(h, smbc, rc[k%2], &sbe[k%bufs], lastsmb, &release);

            output_task (h, oc, &sbe[k%bufs]);            

            if (h->controlfps ){
                struct timespec curtime, difftime;
                int64_t curtime_ns, difftime_ns;
                clock_gettime(CLOCK_REALTIME, &curtime);
                curtime_ns = curtime.tv_sec * 1000000000 + curtime.tv_nsec;
                difftime_ns = h->nexttime - curtime_ns;
                h->nexttime = h->nexttime + h->frametime;
                if (difftime_ns> 0){
                    difftime.tv_sec  = difftime_ns/1000000000;
                    difftime.tv_nsec = difftime_ns%1000000000;
                    nanosleep(&difftime, NULL);
                }
            }
        }

        for (int i=0; i< num_pre_ed; i++){
            k++;
            init_ref_list_and_get_dpb_task(h, rc[k%2], &sbe[k%bufs], &init);
            smbc = acquire_smbc(h);
            SuperMBTask *lastsmb= add_decode_slice_3dwave_tasks(rc[k%2], &sbe[k%bufs], smbc);
            release_ref_list_task(h, smbc, rc[k%2], &sbe[k%bufs], lastsmb, &release);

            #pragma omp task inout (*oc) input(sbe[k%bufs])
            output_task (h, oc, &sbe[k%bufs]);
        }

    } else {
        while(!pc->final_frame && frames++ < h->num_frames && !h->quit){
            #pragma omp taskwait on(sbe[k%bufs])
            parse_task( h, pc, nc, &sbe[k%bufs] ); //not task anymore            
            if ( h->ompss_rt && sbe[k%bufs].droppable ) {
                //unset droppable to prevent triggering regular task dropping logic
                sbe[k%bufs].droppable = 0;
                //these tasks can be freely dropped together using ompss rt pragmas                                
                {
                    #pragma omp task inout(*ec[k%bufs]) inout(sbe[k%bufs])
                    decode_slice_entropy_task(h, ec[k%bufs], &sbe[k%bufs]);

                    #pragma omp task inout(*rc[0]) inout(sbe[k%bufs])
                    decode_slice_mb_task(h, rc[0], &sbe[k%bufs]);

                    #pragma omp task inout (*oc) input(sbe[k%bufs])
                    output_task (h, oc, &sbe[k%bufs]);
                }
            } else {
                #pragma omp task inout(*ec[k%bufs]) inout(sbe[k%bufs])
                decode_slice_entropy_task(h, ec[k%bufs], &sbe[k%bufs]);

                #pragma omp task inout(*rc[0]) inout(sbe[k%bufs])
                decode_slice_mb_task(h, rc[0], &sbe[k%bufs]);

                #pragma omp task inout (*oc) input(sbe[k%bufs])
                output_task (h, oc, &sbe[k%bufs]);
            }
            k++;


            if (h->controlfps ){
                struct timespec curtime, difftime;
                int64_t curtime_ns, difftime_ns;
                clock_gettime(CLOCK_REALTIME, &curtime);
                curtime_ns = curtime.tv_sec * 1000000000 + curtime.tv_nsec;
                difftime_ns = h->nexttime - curtime_ns;
                h->nexttime = h->nexttime + h->frametime;
                if (difftime_ns> 0){
                    difftime.tv_sec  = difftime_ns/1000000000;
                    difftime.tv_nsec = difftime_ns%1000000000;
                    nanosleep(&difftime, NULL);
                }
            }
        }
    }
    
    #pragma omp taskwait

    while ((out=output_frame(h, oc, NULL, h->ofile, h->frame_width, h->frame_height))) ;

    print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 1, h->verbose);
    h->num_frames = oc->frame_number;
    /* finished ! */

    free_parse_context(pc);
    free_nal_context  (nc);
    free_output_context(oc);
    for (int i=0; i<bufs; i++){
        free_sb_entry(&sbe[i]);
        free_entropy_context(ec[i]);
    }
    av_free(sbe);

    for (int i=0; i<2; i++){
        free_mbrec_context(rc[i]);
    }

#if HAVE_LIBSDL2
    if (h->display){
        signal_sdl_exit(h);
        pthread_join(sdl_thr, NULL);
    }
#endif

    return 0;
}
