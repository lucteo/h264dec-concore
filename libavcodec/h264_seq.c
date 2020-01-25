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
// #undef NDEBUG
#include <assert.h>

static int decode_slice_entropy_seq(H264Context *h, EntropyContext *ec, H264Slice *s, GetBitContext *gb, H264Mb *mbs){
    int i,j;
//     GetBitContext *gb = s->gb;
    CABACContext *c = &ec->c;

    if( !s->pps.cabac ){
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return -1;
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
            //memset(m, 0, sizeof(H264Mb));
            m->mb_x=i;
            m->mb_y=j;
            ec->m = m;

            ret = ff_h264_decode_mb_cabac(ec, s, c);
            eos = get_cabac_terminate( c);
            (void) eos;
            if( ret < 0 || c->bytestream > c->bytestream_end + 2) {
                av_log(AV_LOG_ERROR, "error while decoding MB %d %d, bytestream (%td)\n", m->mb_x, m->mb_y, c->bytestream_end - c->bytestream);
                return -1;
            }
        }
    }

//     av_freep(&s->gb.raw);
//     if (s->gb.rbsp)
//         av_freep(&s->gb.rbsp);

    return 0;
}



/**
*   Sequential version
*/
static void decode_slice_mb_seq(H264Context *h, MBRecContext *d, H264Slice *s2, H264Mb *mbs){

    for (int i=0; i<2; i++){
        for(int j=0; j< s2->ref_count[i]; j++){
            if (s2->ref_list_cpn[i][j] ==-1)
                continue;
            int k;
            for (k=0; k<h->max_dpb_cnt; k++){
                if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == s2->ref_list_cpn[i][j]){
                    s2->dp_ref_list[i][j] = &h->dpb[k];
                    break;
                }
            }
        }
    }

    get_dpb_entry(h, s2);

    if (!h->no_mbd){
        for(int j=0; j<d->mb_height; j++){
            init_mbrec_context(d, d->mrs, s2, j);
            if (h->profile) printf("\n[MBREC LINE %d ", j);
            for(int i=0; i<d->mb_width; i++){

                if ((i & 0x7) == 0) start_timer(h, REC);
                H264Mb *m = &mbs[i + j*d->mb_width];
                if (h->profile==2)
                    pred_motion_mb_rec (d, d->mrs, s2, m);
                else{
                    h264_decode_mb_internal(d, d->mrs, s2, m);
                }
                stop_timer(h, REC);
            }
            draw_edges(d, s2, j);

        }
    }

    for (int i=0; i<s2->release_cnt; i++){
        for(int j=0; j<h->max_dpb_cnt; j++){
            if(h->dpb[j].cpn== s2->release_ref_cpn[i]){
                release_dpb_entry(h, &h->dpb[j], 2);
                break;
            }
        }
    }
    s2->release_cnt=0;
}

/*
* The following code is the main loop of the file converter
*/
int h264_decode_seq( H264Context *h) {
    ParserContext *pc;
    NalContext *nc;
    EntropyContext *ec;
    MBRecContext *rc;
    OutputContext *oc;

    H264Slice slice, *s=&slice;
    H264Mb *mbs;
    DecodedPicture *out;
    int frames=0;

#if HAVE_LIBSDL2
    pthread_t sdl_thr;
    if (h->display){
        pthread_create(&sdl_thr, NULL, sdl_thread, h);
    }
#endif

    pc = get_parse_context(h->ifile);
    nc = get_nal_context(h->width, h->height);

    memset(s, 0, sizeof(H264Slice));
    mbs = av_malloc( h->mb_height * h->mb_width * sizeof(H264Mb));

    ec = get_entropy_context( h );
    rc = get_mbrec_context(h);
    rc->top_next = rc->top = av_malloc( h->mb_width * sizeof(TopBorder));

    oc = get_output_context( h );

    av_start_timer();
    GetBitContext gb = {0,};
    while(!pc->final_frame && frames++ < h->num_frames && !h->quit){
        if (h->profile) start_timer(h, FRONT);
        av_read_frame_internal(pc, &gb);
        decode_nal_units(nc, s, &gb);
        if (h->profile) stop_timer(h, FRONT);
//         memset(s->mbs, 0, sizeof(H264Mb)*ec->mb_width*ec->mb_height);
        if (h->profile) start_timer(h, ED);
        decode_slice_entropy_seq(h, ec, s, &gb, mbs);
        if (h->profile) stop_timer(h, ED);

        if (h->profile) start_timer(h, REC);
        decode_slice_mb_seq(h, rc, s, mbs);
        if (h->profile) stop_timer(h, REC);

        out =output_frame(h, oc, s->curr_pic, h->ofile, h->frame_width, h->frame_height);
        if (out){
            release_dpb_entry(h, out, 1);
        }

        print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 0, h->verbose);
        if (h->profile == 3){
            printf("[ENTROPY %.3fms] [MBREC %.3fms]\n", h->last_time[ED] , h->last_time[REC]);
        }
    }
    while ((out=output_frame(h, oc, NULL, h->ofile, h->frame_width, h->frame_height))) ;

    print_report(oc->frame_number, oc->dropped_frames, oc->video_size, 1, h->verbose);
    h->num_frames = oc->frame_number;
    /* finished ! */
    av_freep(&mbs);
    av_freep(&gb.raw);
    if (gb.rbsp)
        av_freep(&gb.rbsp);
    av_freep(&rc->top);

    free_parse_context(pc);
    free_nal_context  (nc);
    free_entropy_context(ec);
    free_mbrec_context(rc);
    free_output_context(oc);

#if HAVE_LIBSDL2
    if (h->display){
        signal_sdl_exit(h);
        pthread_join(sdl_thr, NULL);
    }
#endif

    return 0;
}
