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
#include "h264_misc.h"
#include "cell/h264_types_spu.h"
#include "h264_pthread.h"

#include <pthread.h>
#include <assert.h>
#include <unistd.h>

#include <libspe2.h>
#include <ppu_intrinsics.h>
#include <cbe_mfc.h>
#include <libsync.h>

// spe global variables
unsigned rl_cnt_var, rl_mutex_var, rl_cond_var;
atomic_ea_t rl_cnt;
cond_ea_t rl_cond;
mutex_ea_t rl_lock;

H264spe * spe_params;
unsigned mutex_var[16];
unsigned cond_var[16];
unsigned atomic_var[16];

pthread_t * spe_tid;
spe_context_ptr_t *spe_context;
void** spe_control_area;
void** spe_ls_area;
H264slice **spe_slice_buf;

H264spe * spe_ed_params;
unsigned mutex_ed_var[16];
unsigned cond_ed_var[16];
unsigned atomic_ed_var[16];

pthread_t * spe_ed_tid;
spe_context_ptr_t *spe_ed_context;
void** spe_ed_control_area;
void** spe_ed_ls_area;
EDSlice_spu **spe_ed_slice_buf;

//structs to propagate stop signal
MBSlice last_slice;
EDSlice last_ed_slice;
DecodedPicture last_pic;
RawFrame last_frm;

static int direct_B_resolved(EDSlice *s, int *poc_list, int *poc_cnt){
    int i;
    int cnt = *poc_cnt;
    for(i=0; i<cnt; i++){
        if (poc_list[i]==s->ref_list[1][0]->poc){
            *poc_cnt=i+1;
            while(++i<cnt)
                poc_list[i]=0;
            return 1;
        }
    }
    return 0;
}

static void update_IP_poc_list(int *poc_list, int *poc_cnt, int poc) {
    int i=0;
    int cnt = *poc_cnt;

    while (poc_list[i] > poc) { i++;}
    if ( i< cnt)
        memmove(&poc_list[i+1], &poc_list[i], (cnt-i)*sizeof(int));

    poc_list[i]=poc;
    (*poc_cnt)++;
}

static void *spe_ed_thread(void *arg){
    H264spe *params = (H264spe *)arg;
    unsigned int idx = params->idx;
    unsigned int runflags = 0;
    unsigned int entry = SPE_DEFAULT_ENTRY;
    // run SPE context
    spe_context_run(spe_ed_context[idx],  &entry, runflags, (void*) params, NULL, NULL);
    // done - now exit thread
    pthread_exit(NULL);
}

static void create_spe_ED_threads(H264Context *h, int ip_threads, int b_threads) {
    int i;
    int num_threads = ip_threads+b_threads;
    spe_program_handle_t * spe_program = spe_image_open("spe_ed");
    // reserve memory for spe thread id, context and argument addresses
    spe_ed_tid = av_malloc(num_threads * sizeof (pthread_t));
    spe_ed_context = av_malloc(num_threads * sizeof (spe_context_ptr_t));
    spe_ed_params = av_malloc(num_threads * sizeof (H264spe));
    spe_ed_control_area = av_malloc(num_threads * sizeof (void*));
    spe_ed_ls_area = av_malloc(num_threads * sizeof (void*));
    spe_ed_slice_buf = av_malloc(num_threads * sizeof (void*));

    if (spe_program == NULL)
        av_log(AV_LOG_ERROR, "PPE: error opening SPE object image:%d. error=%s \n", errno, strerror(errno));

    for (i = 0; i < num_threads; i++) {
        // create context for spe program
        spe_ed_context[i] = spe_context_create(SPE_MAP_PS, NULL);
        if (spe_ed_context[i] == NULL)
            av_log(AV_LOG_ERROR, "PPE: error creating SPE context:%d. error=%s \n", errno, strerror(errno));
        // load SPE program into main memory
        if ((spe_program_load(spe_ed_context[i], spe_program)) == -1)
            av_log(AV_LOG_ERROR, "PPE: error loading SPE context:%d. error=%s \n", errno, strerror(errno));
        //get the control_area for fast mailboxing
        if ((spe_ed_control_area[i] = spe_ps_area_get(spe_ed_context[i], SPE_CONTROL_AREA)) == NULL)
            av_log(AV_LOG_ERROR, "PPE: error retrieving SPE control area:%d. error=%s \n", errno, strerror(errno));
        //get ls area for inter spe communication
        if ((spe_ed_ls_area[i] = spe_ls_area_get(spe_ed_context[i])) == NULL)
            av_log(AV_LOG_ERROR, "PPE: error retrieving SPE ls area:%d. error=%s \n", errno, strerror(errno));
    }

    for (i = 0; i < ip_threads; i++) {
        spe_ed_params[i].mb_width = h->mb_width;
        spe_ed_params[i].mb_stride = h->mb_stride;
        spe_ed_params[i].mb_height = h->mb_height;
        spe_ed_params[i].type = EDIP;
        spe_ed_params[i].spe_id = i;
        spe_ed_params[i].idx = i;
        //spe_ed_params[i].spe_total = ip_threads; //not used
        //spe_params[i].slice_params= &slice_params;
        spe_ed_params[i].src_spe = spe_ed_ls_area[(i-1+num_threads)%num_threads];
        spe_ed_params[i].tgt_spe = spe_ed_ls_area[(i+1)%num_threads];

        spe_ed_params[i].lock = (mutex_ea_t) (unsigned) &mutex_ed_var[i];
        spe_ed_params[i].cond = (cond_ea_t) (unsigned) &cond_ed_var[i];
        spe_ed_params[i].cnt = (atomic_ea_t)(unsigned) &atomic_ed_var[i]; atomic_set(spe_ed_params[i].cnt, 0);

        mutex_init(spe_ed_params[i].lock);
        cond_init(spe_ed_params[i].cond);
        if (pthread_create(&spe_ed_tid[i], NULL, spe_ed_thread, (void *) &spe_ed_params[i]))
            av_log(AV_LOG_ERROR, "create_workers: pthread create for spe failed %d\n", i);

        //slicebufaddr
        spe_ed_slice_buf[i] = (EDSlice_spu *) _spe_out_mbox_read(spe_ed_control_area[i]);
        av_log(AV_LOG_DEBUG, "create_workers: created spe thread %d\n", i);
    }
    for (int j = 0; j < b_threads; j++) {
        i = j+ip_threads;
        spe_ed_params[i].mb_width = h->mb_width;
        spe_ed_params[i].mb_stride = h->mb_stride;
        spe_ed_params[i].mb_height = h->mb_height;
        spe_ed_params[i].type = EDB;
        spe_ed_params[i].idx = i;
        spe_ed_params[i].spe_id = j;
        spe_ed_params[i].spe_total = b_threads;
        //spe_params[i].slice_params= &slice_params;
        //spe_ed_params[i].src_spe = spe_ed_ls_area[(i-1+num_threads)%num_threads];
        spe_ed_params[i].tgt_spe = spe_ed_ls_area[((j+1)%b_threads) + ip_threads];

        spe_ed_params[i].lock = (mutex_ea_t) (unsigned) &mutex_ed_var[i];
        spe_ed_params[i].cond = (cond_ea_t) (unsigned) &cond_ed_var[i];
        spe_ed_params[i].cnt = (atomic_ea_t)(unsigned) &atomic_ed_var[i]; atomic_set(spe_ed_params[i].cnt, 0);

        mutex_init(spe_ed_params[i].lock);
        cond_init(spe_ed_params[i].cond);
        if (pthread_create(&spe_ed_tid[i], NULL, spe_ed_thread, (void *) &spe_ed_params[i]))
            av_log(AV_LOG_ERROR, "create_workers: pthread create for spe failed %d\n", i);

        //slicebufaddr
        spe_ed_slice_buf[i] = (EDSlice_spu *) _spe_out_mbox_read(spe_ed_control_area[i]);
        av_log(AV_LOG_DEBUG, "create_workers: created spe thread %d\n", i);
    }
    spe_image_close(spe_program);

}

static void fill_EDSlice_spu(EDSlice_spu *dst, EDSlice *src){
    dst->pps 	= src->pps;
    dst->mbs 	= src->mbs;
    dst->state 	= src->state;
    dst->qp_thresh = src->qp_thresh;
    dst->pic	= *src->current_picture;

    dst->ref_count[0] = src->ref_count[0];
    dst->ref_count[1] = src->ref_count[1];
    dst->slice_type	  = src->slice_type;
    dst->slice_type_nos = src->slice_type_nos;
    dst->direct_8x8_inference_flag = src->direct_8x8_inference_flag;
    dst->list_count = src->list_count;
    dst->coded_pic_num = src->coded_pic_num;

    GetBitContext *gb = &src->gb;
    align_get_bits( gb);
    dst->bytestream_start = gb->buffer + get_bits_count(gb)/8;
    dst->byte_bufsize = (get_bits_left(gb) + 7)/8;

    dst->transform_bypass = src->transform_bypass;
    dst->direct_spatial_mv_pred = src->direct_spatial_mv_pred;
    memcpy(dst->map_col_to_list0, src->map_col_to_list0, 2*16*sizeof(int));
    memcpy(dst->dist_scale_factor, src->dist_scale_factor, 16*sizeof(int));
    dst->cabac_init_idc = src->cabac_init_idc;
    memcpy(dst->ref2frm, src->ref2frm, 2*64*sizeof(int));
    dst->chroma_qp[0]= src->chroma_qp[0];
    dst->chroma_qp[1]= src->chroma_qp[1];
    dst->qscale = src->qscale;
    dst->last_qscale_diff = src->last_qscale_diff;

    if (src->slice_type_nos == FF_B_TYPE) dst->list1 = *src->ref_list[1][0];
}

static void send_slice_to_spe_and_wait(EDSlice_spu *s, int id){
    unsigned status;

    spe_mfcio_get(spe_ed_context[id], (unsigned) spe_ed_slice_buf[id], s, sizeof(EDSlice_spu), 14, 0, 0);
    spe_mfcio_tag_status_read(spe_ed_context[id], 1<<14, SPE_TAG_ALL, &status);


    _spe_in_mbox_write(spe_ed_control_area[id], 0);

    while (!spe_out_mbox_status(spe_ed_context[id])){
        //pthread_yield();
        usleep(1000);
    }
    _spe_out_mbox_read(spe_ed_control_area[id]);
}

static int decode_slice_entropy_cell(EntropyContext *ec, EDSlice *s, int id){
    int i,j;

    if( !s->pps.cabac ){
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return -1;
    }
    DECLARE_ALIGNED(16, EDSlice_spu, slice);
    fill_EDSlice_spu(&slice, s);

    send_slice_to_spe_and_wait(&slice, id);

    return 0;
}

static int decode_slice_entropy_cell_seq(H264Context *h, EntropyContext *ec, EDSlice *s){
    int i,j;

    if( !s->pps.cabac ){
        av_log(AV_LOG_ERROR, "Only cabac encoded streams are supported\n");
        return -1;
    }
    DECLARE_ALIGNED(16, EDSlice_spu, slice);
    fill_EDSlice_spu(&slice, s);

    send_slice_to_spe_and_wait(&slice, 0);

    if (s->release_cnt>0) {
        for (int i=0; i<s->release_cnt; i++){
            release_pib_entry(h, s->release_ref[i], 2);
        }
        s->release_cnt=0;
    }

    release_pib_entry(h, s->current_picture, 1);
    av_freep(&s->gb.raw);
    if (s->gb.rbsp)
        av_freep(&s->gb.rbsp);

    return 0;
}

static void *entr_IP_spe_thread(void *arg){
    EDThreadContext *eip = (EDThreadContext *) arg;
    H264Context *h = eip->h;
// 	printf("eip %d, pid %d\n", eip->thread_num, syscall(SYS_gettid));
    for (int i=0; i<SLICE_BUFS; i++){
        eip->mbs[i] = av_malloc(h->mb_height*h->mb_width*sizeof(H264Mb));
    }

    EntropyContext *ec = get_entropy_context(h);
    EDSlice *s;

    for(;;){
        {
            pthread_mutex_lock(&eip->ed_lock);
            while (eip->ed_cnt <= 0)
                pthread_cond_wait(&eip->ed_cond, &eip->ed_lock);
            s = &eip->ed_q[eip->ed_fo];
            eip->ed_fo++; eip->ed_fo %= MAX_SLICE_COUNT;
            pthread_mutex_unlock(&eip->ed_lock);
        }

        if (s->state<0)
            break;
        {
            pthread_mutex_lock(&eip->mbs_lock);
            while (eip->mbs_cnt <= 0)
                pthread_cond_wait(&eip->mbs_cond, &eip->mbs_lock);

            s->mbs = eip->mbs[eip->mbs_fo];
            s->ed = eip;
            eip->mbs_cnt--;
            eip->mbs_fo++; eip->mbs_fo%=SLICE_BUFS;
            pthread_mutex_unlock(&eip->mbs_lock);
        }
        if (eip->cell){
            decode_slice_entropy_cell(ec, s, eip->thread_num);
        }else{
            decode_slice_entropy(ec, s);
        }

//         {
//             pthread_mutex_lock(&h->lock[ENTROPY2]);
//             h->ed_poc[h->ed_poc_fi++ % MAX_SLICE_COUNT] = s->current_picture->poc;
//             while (h->ed_poc_fi > h->ed_poc_fo + MAX_SLICE_COUNT)
//                 h->ed_poc_fo++;
//
//             pthread_cond_signal(&h->cond[ENTROPY2]);
//             pthread_mutex_unlock(&h->lock[ENTROPY2]);
//         }

        {
            pthread_mutex_lock(&h->lock[ENTROPY4]);
            while (h->ed_reorder_cnt>=MAX_SLICE_COUNT)
                pthread_cond_wait(&h->cond[ENTROPY4], &h->lock[ENTROPY4]);
            h->ed_reorder_q[h->ed_reorder_fi] = *s;
            h->ed_reorder_cnt++;
            h->ed_reorder_fi++; h->ed_reorder_fi %= MAX_SLICE_COUNT;
            pthread_cond_signal(&h->cond[ENTROPY4]);
            pthread_mutex_unlock(&h->lock[ENTROPY4]);
        }

        {
            pthread_mutex_lock(&eip->ed_lock);
            eip->ed_cnt--;
            pthread_cond_signal(&eip->ed_cond);
            pthread_mutex_unlock(&eip->ed_lock);
        }
    }

    free_entropy_context(ec);

    pthread_exit(NULL);
    return NULL;
}

static void *entr_B_spe_thread(void *arg){
    EDThreadContext *eb = (EDThreadContext *) arg;
    H264Context *h = eb->h;
// 	printf("eb %d, pid %d\n", eb->thread_num, syscall(SYS_gettid));
    for (int i=0; i<SLICE_BUFS; i++){
        eb->mbs[i] = av_malloc(h->mb_height*h->mb_width*sizeof(H264Mb));
    }

    EntropyContext *ec = get_entropy_context(h);
    EDSlice *s;

    for(;;){
        {
            pthread_mutex_lock(&eb->ed_lock);
            while (eb->ed_cnt <= 0)
                pthread_cond_wait(&eb->ed_cond, &eb->ed_lock);
            s = &eb->ed_q[eb->ed_fo];
            eb->ed_fo++; eb->ed_fo %= MAX_SLICE_COUNT;
            pthread_mutex_unlock(&eb->ed_lock);
        }

        if (s->state<0)
            break;
        {
            pthread_mutex_lock(&eb->mbs_lock);
            while (eb->mbs_cnt <= 0)
                pthread_cond_wait(&eb->mbs_cond, &eb->mbs_lock);
            s->mbs = eb->mbs[eb->mbs_fo];
            s->ed = eb;
            eb->mbs_cnt--;
            eb->mbs_fo++; eb->mbs_fo%=SLICE_BUFS;
            pthread_mutex_unlock(&eb->mbs_lock);
        }
        //decode_B_slice_entropy(&hcabac, &cabac, s, eb, eb->prev_ed);
        decode_slice_entropy_cell(ec, s, eb->thread_num + h->edip_threads);

        {
            pthread_mutex_lock(&h->lock[ENTROPY4]);
            while (h->ed_reorder_cnt>=MAX_SLICE_COUNT)
                pthread_cond_wait(&h->cond[ENTROPY4], &h->lock[ENTROPY4]);
            h->ed_reorder_q[h->ed_reorder_fi] = *s;
            h->ed_reorder_cnt++;
            h->ed_reorder_fi++; h->ed_reorder_fi %= MAX_SLICE_COUNT;
            pthread_cond_signal(&h->cond[ENTROPY4]);
            pthread_mutex_unlock(&h->lock[ENTROPY4]);

        }

        {
            pthread_mutex_lock(&eb->ed_lock);
            eb->ed_cnt--;
            pthread_cond_signal(&eb->ed_cond);
            pthread_mutex_unlock(&eb->ed_lock);
        }
    }
    eb->lines_cnt++;

    free_entropy_context(ec);

    pthread_exit(NULL);
    return NULL;
}

static void *entr_B_distribute(void *arg){
    H264Context *h = (H264Context *) arg;
    EDSlice *s;

    int i, n=0, poc;

// 	printf("eb dist, pid %d\n", syscall(SYS_gettid));

    for(i=0; i<h->edb_threads; i++){
        h->b[i].h =h;
        h->b[i].thread_num =i;
        h->b[i].thread_total =h->edb_threads;
        pthread_mutex_init(&h->b[i].mbs_lock, NULL);
        pthread_cond_init(&h->b[i].mbs_cond, NULL);
        h->b[i].mbs_fo = 0;
        h->b[i].mbs_cnt = SLICE_BUFS;
        h->b[i].ed_fi =0;
        h->b[i].ed_fo =0;
        h->b[i].ed_cnt =0;
        h->b[i].lines_cnt =0;
        h->b[i].prev_ed = &h->b[(i-1 +h->edb_threads) % h->edb_threads];
        pthread_mutex_init(&h->b[i].ed_lock, NULL);
        pthread_cond_init(&h->b[i].ed_cond, NULL);
        pthread_create(&h->ed_B_thr[i], NULL, entr_B_spe_thread, &h->b[i]);
    }

    for(;;){
        {
            pthread_mutex_lock(&h->lock[ENTROPY3B]);
            while (h->ed_B_cnt<=0)
                pthread_cond_wait(&h->cond[ENTROPY3B], &h->lock[ENTROPY3B]);
            s= &h->ed_B_q[h->ed_B_fo];
            h->ed_B_fo++; h->ed_B_fo %= MAX_SLICE_COUNT;
            pthread_mutex_unlock(&h->lock[ENTROPY3B]);

        }
        if (s->state<0)
            break;

        if (s->ref_list[1][0]->slice_type_nos != FF_B_TYPE){
            while (poc < s->ref_list[1][0]->poc){
                pthread_mutex_lock(&h->lock[ENTROPY2]);
                while (poc == h->ed_poc)
                    pthread_cond_wait(&h->cond[ENTROPY2], &h->lock[ENTROPY2]);
                poc = h->ed_poc;
                pthread_mutex_unlock(&h->lock[ENTROPY2]);
            }
        }
        {
            pthread_mutex_lock(&h->b[n].ed_lock);
            while (h->b[n].ed_cnt >= MAX_SLICE_COUNT)
                pthread_cond_wait(&h->b[n].ed_cond, &h->b[n].ed_lock);
            h->b[n].ed_q[ h->b[n].ed_fi] = *s;
            h->b[n].ed_cnt++;
            h->b[n].ed_fi++; h->b[n].ed_fi %= MAX_SLICE_COUNT;
            pthread_cond_signal(&h->b[n].ed_cond);
            pthread_mutex_unlock(&h->b[n].ed_lock);

            n++; n%=h->edb_threads;
        }
        {
            pthread_mutex_lock(&h->lock[ENTROPY3B]);
            h->ed_B_cnt--;
            pthread_cond_signal(&h->cond[ENTROPY3B]);
            pthread_mutex_unlock(&h->lock[ENTROPY3B]);

        }

    }

    for (i=0; i<h->edb_threads; i++){
        pthread_mutex_lock(&h->b[i].ed_lock);
        while (h->b[i].ed_cnt >= MAX_SLICE_COUNT)
            pthread_cond_wait(&h->b[i].ed_cond, &h->b[i].ed_lock);
        h->b[i].ed_q[ h->b[i].ed_fi] = *s;
        h->b[i].ed_cnt++;
        h->b[i].ed_fi++; h->b[i].ed_fi %= MAX_SLICE_COUNT;
        pthread_cond_signal(&h->b[i].ed_cond);
        pthread_mutex_unlock(&h->b[i].ed_lock);

    }
    for(int i=0; i<h->edb_threads; i++){
        pthread_join(h->ed_B_thr[i], NULL);
    }
    pthread_exit(NULL);
    return NULL;
}


static void *entr_IPB_distribute(void *arg){
    H264Context *h = (H264Context *) arg;
    EDSlice *s;
    int i,n=0;

    create_spe_ED_threads(h, h->edip_threads, h->edb_threads);
    pthread_create(&h->ed_B_dist, NULL, entr_B_distribute, h);
    for(i=0; i<h->edip_threads + h->edip_ppe_threads; i++){
        h->ip[i].h =h;
        h->ip[i].cell = (i >= h->edip_ppe_threads);
        pthread_mutex_init(&h->ip[i].mbs_lock, NULL);
        pthread_cond_init(&h->ip[i].mbs_cond, NULL);
        h->ip[i].thread_num = i - h->edip_ppe_threads;
        h->ip[i].thread_total=h->edip_threads+ h->edip_ppe_threads;
        h->ip[i].mbs_fo = 0;
        h->ip[i].mbs_cnt = SLICE_BUFS;
        h->ip[i].ed_fi =0;
        h->ip[i].ed_fo =0;
        pthread_mutex_init(&h->ip[i].ed_lock, NULL);
        pthread_cond_init(&h->ip[i].ed_cond, NULL);
        pthread_create(&h->ed_IP_thr[i], NULL, entr_IP_spe_thread, &h->ip[i]);
    }

    for(;;){
        {
            pthread_mutex_lock(&h->lock[ENTROPY]);
            while (h->ed_cnt<=0)
                pthread_cond_wait(&h->cond[ENTROPY], &h->lock[ENTROPY]);
            s= &h->ed_q[h->ed_fo];

            pthread_mutex_unlock(&h->lock[ENTROPY]);
            h->ed_fo++; h->ed_fo %= MAX_SLICE_COUNT;
        }
        if (s->state<0)
            break;

        assert(s->current_picture);
        if (s->slice_type_nos == FF_B_TYPE )
        {
            pthread_mutex_lock(&h->lock[ENTROPY3B]);
            while (h->ed_B_cnt>=MAX_SLICE_COUNT)
                pthread_cond_wait(&h->cond[ENTROPY3B], &h->lock[ENTROPY3B]);
            h->ed_B_q[h->ed_B_fi] = *s;
            h->ed_B_cnt++;
            h->ed_B_fi++; h->ed_B_fi %= MAX_SLICE_COUNT;
            pthread_cond_signal(&h->cond[ENTROPY3B]);
            pthread_mutex_unlock(&h->lock[ENTROPY3B]);
        }else
        {
            ///round robin now, change to based on rawframes size.
            pthread_mutex_lock(&h->ip[n].ed_lock);
            while (h->ip[n].ed_cnt >= MAX_SLICE_COUNT)
                pthread_cond_wait(&h->ip[n].ed_cond, &h->ip[n].ed_lock);
            h->ip[n].ed_q[ h->ip[n].ed_fi] = *s;
            h->ip[n].ed_cnt++;
            h->ip[n].ed_fi++; h->ip[n].ed_fi %= MAX_SLICE_COUNT;
            pthread_cond_signal(&h->ip[n].ed_cond);
            pthread_mutex_unlock(&h->ip[n].ed_lock);

            n++; n %=(h->edip_threads+h->edip_ppe_threads);
        }
        {
            pthread_mutex_lock(&h->lock[ENTROPY]);
            h->ed_cnt--;
            pthread_cond_signal(&h->cond[ENTROPY]);
            pthread_mutex_unlock(&h->lock[ENTROPY]);

        }
    }

    {
        pthread_mutex_lock(&h->lock[ENTROPY3B]);
        while (h->ed_B_cnt>=MAX_SLICE_COUNT)
            pthread_cond_wait(&h->cond[ENTROPY3B], &h->lock[ENTROPY3B]);
        h->ed_B_q[h->ed_B_fi] = *s;
        h->ed_B_cnt++;
        h->ed_B_fi++; h->ed_B_fi %= MAX_SLICE_COUNT;
        pthread_cond_signal(&h->cond[ENTROPY3B]);
        pthread_mutex_unlock(&h->lock[ENTROPY3B]);
    }
    {
        for (i=0; i<h->edip_threads + h->edip_ppe_threads; i++){
            pthread_mutex_lock(&h->ip[i].ed_lock);
            while (h->ip[i].ed_cnt >= MAX_SLICE_COUNT)
                pthread_cond_wait(&h->ip[i].ed_cond, &h->ip[i].ed_lock);
            h->ip[i].ed_q[ h->ip[i].ed_fi] = *s;
            h->ip[i].ed_cnt++;
            h->ip[i].ed_fi++; h->ip[i].ed_fi %= MAX_SLICE_COUNT;
            pthread_cond_signal(&h->ip[i].ed_cond);
            pthread_mutex_unlock(&h->ip[i].ed_lock);
        }
    }
    {
        pthread_mutex_lock(&h->lock[ENTROPY4]);
        while (h->ed_reorder_cnt>=MAX_SLICE_COUNT)
            pthread_cond_wait(&h->cond[ENTROPY4], &h->lock[ENTROPY4]);
        h->ed_reorder_q[h->ed_reorder_fi] = *s;
        h->ed_reorder_cnt++;
        h->ed_reorder_fi++; h->ed_reorder_fi %= MAX_SLICE_COUNT;
        pthread_cond_signal(&h->cond[ENTROPY4]);
        pthread_mutex_unlock(&h->lock[ENTROPY4]);

    }
    pthread_join(h->ed_B_dist, NULL);
    for(i=0; i<h->edip_threads; i++){
        pthread_join(h->ed_IP_thr[i], NULL);
    }
    pthread_exit(NULL);
    return NULL;
}

static pthread_t ed_IPB_dist;
static void *entropy_IPB_cell_thread(void *arg){
    H264Context *h = (H264Context *) arg;
    int i;
    EDSlice reorder[MAX_SLICE_COUNT];
    int ip_poc[MAX_SLICE_COUNT][2]={0,};
    int next_ip_id=0;
    int ip_poc_cnt=0;
    EDSlice *s;
    int reorder_cnt=0;
    unsigned next_pic_num=0;

    pthread_create(&ed_IPB_dist, NULL, entr_IPB_distribute, h);
    int count =0;
    for(;;){
        //signals received from the entropy decoders
        {
            pthread_mutex_lock(&h->lock[ENTROPY4]);
            while (h->ed_reorder_cnt<=0)
                pthread_cond_wait(&h->cond[ENTROPY4], &h->lock[ENTROPY4]);
            s= &h->ed_reorder_q[h->ed_reorder_fo];
            h->ed_reorder_fo++; h->ed_reorder_fo %=MAX_SLICE_COUNT;
            pthread_mutex_unlock(&h->lock[ENTROPY4]);
        }

        if (s->state >=0 && s->slice_type_nos != FF_B_TYPE){
            for (i=0; i<ip_poc_cnt; i++){
                if (s->ip_id < ip_poc[i][0]){
                    memmove(ip_poc[i+1], ip_poc[i], 2*(ip_poc_cnt-i)*sizeof(int));
                    break;
                }
            }
            ip_poc[i][0]= s->ip_id;
            ip_poc[i][1]= s->current_picture->poc;
            ip_poc_cnt++;

            while (next_ip_id == ip_poc[0][0]){
                pthread_mutex_lock(&h->lock[ENTROPY2]);
                h->ed_poc = ip_poc[0][1];

                pthread_cond_signal(&h->cond[ENTROPY2]);
                pthread_mutex_unlock(&h->lock[ENTROPY2]);
                memmove(ip_poc[0], ip_poc[1], 2*(ip_poc_cnt-1)*sizeof(int));
                ip_poc_cnt--;
                next_ip_id++;
            }
        }

        for(i=reorder_cnt; i>0; i--){
            if (s->coded_pic_num < reorder[i-1].coded_pic_num)
                break;
            reorder[i]=reorder[i-1];
        }
        reorder[i]=*s;

        while(reorder_cnt>=0){
            if (next_pic_num!=reorder[reorder_cnt].coded_pic_num){
                break;
            }
            EDSlice *es = &reorder[reorder_cnt];

            {
                pthread_mutex_lock(&h->lock[MBDEC]);
                while (h->mbdec_cnt >= MAX_SLICE_COUNT)
                    pthread_cond_wait(&h->cond[MBDEC], &h->lock[MBDEC]);
                copyEDtoMBSlice(&h->mbdec_q[h->mbdec_fi], es);

                h->mbdec_cnt++;
                h->mbdec_fi++; h->mbdec_fi %= MAX_SLICE_COUNT;
                pthread_cond_signal(&h->cond[MBDEC]);
                pthread_mutex_unlock(&h->lock[MBDEC]);

            }

            if (es->state<0)
                goto end;

            assert(es->current_picture);
            for (int i=0; i<es->release_cnt; i++){
                release_pib_entry(h, es->release_ref[i], 2);
            }
            release_pib_entry(h, es->current_picture, 1);
            av_freep(&es->gb.raw);
            if (es->gb.rbsp)
                av_freep(&es->gb.rbsp);

            next_pic_num++;
            reorder_cnt--;
        }
        reorder_cnt++;

        {
            pthread_mutex_lock(&h->lock[ENTROPY4]);
            h->ed_reorder_cnt--;
            pthread_cond_signal(&h->cond[ENTROPY4]);
            pthread_mutex_unlock(&h->lock[ENTROPY4]);
        }
    }

end:
    pthread_join(ed_IPB_dist, NULL);
    pthread_exit(NULL);
    return NULL;
}


static void fill_spe_slice(H264slice *dst, const MBSlice *src, H264Context *h){
    dst->deblocking_filter =1;
    dst->linesize = src->current_picture->linesize[0];
    dst->uvlinesize = src->current_picture->linesize[1];
    dst->mb_width = h->mb_width;
    dst->mb_height = h->mb_height;
    dst->use_weight = src->use_weight;
    dst->use_weight_chroma = src->use_weight_chroma;
    dst->luma_log2_weight_denom = src->luma_log2_weight_denom;
    dst->chroma_log2_weight_denom = src->chroma_log2_weight_denom;

    //weights later
    memcpy(dst->luma_weight, src->luma_weight, 16*2*2*sizeof(int16_t));
    memcpy(dst->chroma_weight, src->chroma_weight, 16*2*2*2*sizeof(int16_t));
    memcpy(dst->implicit_weight, src->implicit_weight, 16*16*2*sizeof(int16_t));

    for(int list=0; list<2; list++){
        for (int i=0; i<src->ref_count[list]; i++){
            Picture_spu *p_dst = &dst->ref_list[list][i];
            DecodedPicture *p_src = src->ref_list[list][i];
            if (p_src){
                p_dst->data[0] = p_src->data[0];
                p_dst->data[1] = p_src->data[1];
                p_dst->data[2] = p_src->data[2];
            }
        }
    }
    dst->state = src->state;

    dst->emu_edge_width  =32;
    dst->emu_edge_height =32;
    dst->slice_type = src->slice_type;
    dst->slice_type_nos = src->slice_type_nos;
    dst->slice_alpha_c0_offset = src->slice_alpha_c0_offset;
    dst->slice_beta_offset = src->slice_beta_offset;

    memcpy(dst->chroma_qp_table, src->pps.chroma_qp_table, 2*64);

    dst->blocks = src->mbs;
    dst->dst_y = src->current_picture->data[0];
    dst->dst_cb = src->current_picture->data[1];
    dst->dst_cr = src->current_picture->data[2];
}

static void decode_slice_mb_seq_cell(H264Context *h, MBRecContext *d, MBSlice *s, DecodedPicture *tmp){
    static int rl_fi=0;

    DECLARE_ALIGNED(16, H264slice, spe_slice);
    H264spe *p=&spe_params[0];
    unsigned status;
    uint8_t *dst_y, *dst_cb, *dst_cr;

    DecodedPicture *dp;

    for (int i=0; i<2; i++){
        for(int j=0; j< s->ref_count[i]; j++){
            if (s->ref_list_cpn[i][j] ==-1)
                continue;
            int k;
            for (k=0; k<DPB_SIZE; k++){
                if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == s->ref_list_cpn[i][j]){
                    s->ref_list[i][j] = &h->dpb[k];
                    break;
                }
            }
        }
    }

    dp = get_dpb_entry(h);
    init_dpb_entry(dp, s, d->width, d->height);

    if (h->no_mbd)
        return;


    fill_spe_slice(&spe_slice, s, h);
    spe_mfcio_get(spe_context[0], (unsigned) (spe_slice_buf[0] + rl_fi), &spe_slice, sizeof(H264slice), 15, 0, 0);
    spe_mfcio_tag_status_read(spe_context[0], 1<<15, SPE_TAG_ALL, &status);
    rl_fi++; rl_fi %= 2;

    _spe_in_mbox_write(spe_control_area[0], 0);
    while (atomic_read(rl_cnt)<=0){
        //pthread_yield();
        usleep(1000);
    }
    atomic_dec(rl_cnt);


/** This is error free, no visual artifacts, however, md5sum fails.... (WTF) **/
// 	memcpy(tmp->data[0], s->current_picture->data[0], tmp->linesize[0]*h->mb_height*16);
// 	memcpy(tmp->data[1], s->current_picture->data[1], tmp->linesize[1]*h->mb_height*8);
// 	memcpy(tmp->data[2], s->current_picture->data[2], tmp->linesize[1]*h->mb_height*8);
//
// 	memset(s->current_picture->data[0], 0, tmp->linesize[0]*h->mb_height*16);
// 	memset(s->current_picture->data[1], 0, tmp->linesize[1]*h->mb_height*8);
// 	memset(s->current_picture->data[2], 0, tmp->linesize[1]*h->mb_height*8);
//
// 	decode_slice_mb_seq(d, s);
//
// 	for (int i=0; i<h->mb_height*16; i++){
// 		for (int j=0; j<h->width; j++){
// 			if (tmp->data[0][j + i*tmp->linesize[0]] != s->current_picture->data[0][j + i*tmp->linesize[0]]){
// 				printf("%d, %d, %d, %d\n", j, i, tmp->data[0][j + i*tmp->linesize[0]], s->current_picture->data[0][j + i*tmp->linesize[0]]);
// 				return;
// 			}
// 		}
// 	}
//
// 	for (int i=0; i<h->mb_height*8; i++){
// 		for (int j=0; j<h->width/2; j++){
// 			if (tmp->data[1][j + i*tmp->linesize[1]] != s->current_picture->data[1][j + i*tmp->linesize[1]]){
// 				printf("%d, %d, %d, %d\n", j, i, tmp->data[1][j + i*tmp->linesize[1]], s->current_picture->data[1][j + i*tmp->linesize[1]]);
// 				return;
// 			}
// 		}
// 	}
//
// 	for (int i=0; i<h->mb_height*8; i++){
// 		for (int j=0; j<h->width/2; j++){
// 			if (tmp->data[2][j + i*tmp->linesize[1]] != s->current_picture->data[2][j + i*tmp->linesize[1]]){
// 				printf("%d, %d, %d, %d\n", j, i, tmp->data[2][j + i*tmp->linesize[1]], s->current_picture->data[2][j + i*tmp->linesize[1]]);
// 				return;
// 			}
// 		}
// 	}


    //printf("dst_y %p\n", dst_y);


     for (int i=0; i<s->release_cnt; i++){
        for(int j=0; j<DPB_SIZE; j++){
            if(h->dpb[j].cpn== s->release_ref_cpn[i]){
                release_dpb_entry(h, &h->dpb[j], 2);
                break;
            }
        }
    }
    s->release_cnt=0;

}

static void *h264_spe_thread(void * thread_args ) {
    H264spe *params = (H264spe *)thread_args;
    unsigned int spe_id = params->spe_id;
    unsigned int runflags = 0;
    unsigned int entry = SPE_DEFAULT_ENTRY;
    // run SPE context
    spe_context_run(spe_context[spe_id],  &entry, runflags, (void*) params, NULL, NULL);
    // done - now exit thread
    pthread_exit(NULL);
}

static int create_spe_MBR_threads(H264Context *h, int num_threads) {
    int i;

    // reserve memory for spe thread id, context and argument addresses
    spe_tid = av_malloc(num_threads * sizeof (pthread_t));
    spe_context = av_malloc(num_threads * sizeof (spe_context_ptr_t));
    spe_params = av_malloc(num_threads * sizeof (H264spe));
    spe_control_area = av_malloc(num_threads * sizeof (void*));
    spe_ls_area = av_malloc(num_threads * sizeof (void*));
    spe_slice_buf = av_malloc(num_threads * sizeof (void*));

    spe_program_handle_t *spe_program = spe_image_open("spe_mbd");

    if (spe_program == NULL)
        av_log(AV_LOG_ERROR, "PPE: error opening SPE object image:%d. error=%s \n", errno, strerror(errno));

    for (i = 0; i < num_threads; i++) {
        // create context for spe program
        spe_context[i] = spe_context_create(SPE_MAP_PS, NULL);
        if (spe_context[i] == NULL)
            av_log(AV_LOG_ERROR, "PPE: error creating SPE context:%d. error=%s \n", errno, strerror(errno));
        // load SPE program into main memory
        if ((spe_program_load(spe_context[i], spe_program)) == -1)
            av_log(AV_LOG_ERROR, "PPE: error loading SPE context:%d. error=%s \n", errno, strerror(errno));
        //get the control_area for fast mailboxing
        if ((spe_control_area[i] = spe_ps_area_get(spe_context[i], SPE_CONTROL_AREA)) == NULL)
            av_log(AV_LOG_ERROR, "PPE: error retrieving SPE control area:%d. error=%s \n", errno, strerror(errno));
        //get ls area for inter spe communication
        if ((spe_ls_area[i] = spe_ls_area_get(spe_context[i])) == NULL)
            av_log(AV_LOG_ERROR, "PPE: error retrieving SPE ls area:%d. error=%s \n", errno, strerror(errno));
    }

    for (i = 0; i < num_threads; i++) {
        spe_params[i].mb_width = h->mb_width;
        spe_params[i].mb_height = h->mb_height;
        spe_params[i].mb_stride = h->mb_stride;
        spe_params[i].spe_id = i;
        spe_params[i].spe_total = num_threads;
        //spe_params[i].slice_params= &slice_params;
        spe_params[i].src_spe = spe_ls_area[(i-1+num_threads)%num_threads];
        spe_params[i].tgt_spe = spe_ls_area[(i+1)%num_threads];

        spe_params[i].rl_lock = rl_lock;
        spe_params[i].rl_cond = rl_cond;
        spe_params[i].rl_cnt = rl_cnt;
        spe_params[i].lock = (mutex_ea_t) (unsigned) &mutex_var[i];
        spe_params[i].cond = (cond_ea_t) (unsigned) &cond_var[i];
        spe_params[i].cnt = (atomic_ea_t)(unsigned) &atomic_var[i]; atomic_set(spe_params[i].cnt, 0);

        mutex_init(spe_params[i].lock);
        cond_init(spe_params[i].cond);
        if (pthread_create(&spe_tid[i], NULL, h264_spe_thread, (void *) &spe_params[i]))
            av_log(AV_LOG_ERROR, "create_workers: pthread create for spe failed %d\n", i);

        //slicebufaddr
        spe_slice_buf[i] = (H264slice *) _spe_out_mbox_read(spe_control_area[i]);

        av_log(AV_LOG_DEBUG, "create_workers: created spe thread %d\n", i);
    }
    spe_image_close(spe_program);
    return 0;
}

//_spe_out_mbox_read(spe_control_area[i]);
/**
* joins all the spe worker threads.
*/
static void join_spe_worker_threads(H264slice *s, int num_threads, int *rl_fi) {
    int i;
    ///just to keep coding consistency.
    {
        for (i=0; i<num_threads; i++){
            H264spe *p=&spe_params[i];
            unsigned status;

            while (atomic_read(p->cnt)>=2) {//double buffered
                usleep(1000);//cond_wait(p->cond, p->lock);
            }

            spe_mfcio_get(spe_context[i], (unsigned) (spe_slice_buf[i] + rl_fi[i]), s, sizeof(H264slice), 15, 0, 0);
            spe_mfcio_tag_status_read(spe_context[i], 1<<15, SPE_TAG_ALL, &status);
            //mutex_unlock(p->lock);
            _spe_in_mbox_write(spe_control_area[i], 0);
        }
    }

    for (i=0; i<num_threads; i++){
        pthread_join(spe_tid[i], NULL);
    }

    for (i=0; i<num_threads; i++){
        spe_context_destroy(spe_context[i]);
    }
    atomic_inc(rl_cnt);

    // destroy memory reserved for spe thread id, context and argument addresses
    av_freep(&spe_tid);
    av_freep(&spe_context);
    av_freep(&spe_params);
    av_freep(&spe_control_area);
    av_freep(&spe_slice_buf);
}


static void *rl_dist_thread(void *arg){
    int i;
    H264Context *h = (H264Context *) arg;
    MBSlice *s;
    DecodedPicture *dp;
    int rl_fi[16]={0,};
    DECLARE_ALIGNED(16, H264slice, spe_slice);

    create_spe_MBR_threads(h, h->rl_threads);
    for(;;){
        {
            pthread_mutex_lock(&h->lock[MBDEC]);
            while (h->mbdec_cnt<=0)
                pthread_cond_wait(&h->cond[MBDEC], &h->lock[MBDEC]);
            s= &h->mbdec_q[h->mbdec_fo];
            h->mbdec_fo++; h->mbdec_fo %= MAX_SLICE_COUNT;
            pthread_mutex_unlock(&h->lock[MBDEC]);
        }

        if (s->state<0){
            break;
        }
        for (int i=0; i<2; i++){
            for(int j=0; j< s->ref_count[i]; j++){
                if (s->ref_list_cpn[i][j] ==-1)
                    continue;
                int k;
                for (k=0; k<DPB_SIZE; k++){
                    if(h->dpb[k].reference >= 2 && h->dpb[k].cpn == s->ref_list_cpn[i][j]){
                        s->ref_list[i][j] = &h->dpb[k];
                        break;
                    }
                }

            }
        }
        dp = get_dpb_entry(h);
        init_dpb_entry(dp, s, h->width, h->height);
        assert(s->current_picture);
        {
            while (atomic_read(rl_cnt) >=MAX_SLICE_COUNT){
                usleep(1000);
            }
            h->mbrel_q[h->mbrel_fi] = *s;

            h->mbrel_fi++; h->mbrel_fi %= MAX_SLICE_COUNT;
        }
        {
            if(h->no_mbd){
                atomic_inc(rl_cnt);
            }else {
                fill_spe_slice(&spe_slice, s, h);
                for (i=0; i<h->rl_threads; i++){
                    H264spe *p=&spe_params[i];
                    unsigned status;
                    while (atomic_read(p->cnt)>=2){ //double buffered
                        usleep(1000);
                        //cond_wait(p->cond, p->lock);
                    }
                    spe_mfcio_get(spe_context[i], (unsigned) (spe_slice_buf[i] + rl_fi[i]), &spe_slice, sizeof(H264slice), 15, 0, 0);
                    spe_mfcio_tag_status_read(spe_context[i], 1<<15, SPE_TAG_ALL, &status);
                    rl_fi[i]++; rl_fi[i] %= 2;
                    atomic_inc(p->cnt);

                    _spe_in_mbox_write(spe_control_area[i], 0);
                }
            }
        }

        {
            pthread_mutex_lock(&h->lock[MBDEC]);
            h->mbdec_cnt--;
            pthread_cond_signal(&h->cond[MBDEC]);
            pthread_mutex_unlock(&h->lock[MBDEC]);
        }

    }

    {
        while (atomic_read(rl_cnt) >=MAX_SLICE_COUNT){
            usleep(1000);
        }
        h->mbrel_q[h->mbrel_fi] = *s;

        h->mbrel_fi++; h->mbrel_fi %= MAX_SLICE_COUNT;
    }
    spe_slice.state=-1;
    join_spe_worker_threads(&spe_slice, h->rl_threads, rl_fi);
    pthread_exit(NULL);
    return NULL;
}

static void *mbdec_cell_thread(void *arg){
    H264Context *h = (H264Context *) arg;

    rl_lock = (mutex_ea_t) (unsigned) &rl_mutex_var;
    rl_cond = (cond_ea_t) (unsigned) &rl_cond_var;
    rl_cnt = (atomic_ea_t) (unsigned) &rl_cnt_var;
    atomic_set(rl_cnt, 0);
    mutex_init(rl_lock);
    cond_init(rl_cond);
// 	printf("mbdec, pid %d\n", syscall(SYS_gettid));
    pthread_create(&h->rl_dist_thr, NULL, rl_dist_thread, h);

    for(;;){
        MBSlice *s=NULL;
        {
            while (atomic_read(rl_cnt)<=0){
                usleep(1000);
            }
            s= &h->mbrel_q[h->mbrel_fo];
            h->mbrel_fo++; h->mbrel_fo %= MAX_SLICE_COUNT;
        }

        if (s->state<0)
            break;

        for (int i=0; i<s->release_cnt; i++){
            for(int j=0; j<DPB_SIZE; j++){
                if(h->dpb[j].cpn== s->release_ref_cpn[i]){
                    release_dpb_entry(h, &h->dpb[j], 2);
                    break;
                }
            }
        }

        {
            EDThreadContext *ed = s->ed;
            pthread_mutex_lock(&ed->mbs_lock);
            ed->mbs_cnt++;
            pthread_cond_signal(&ed->mbs_cond);
            pthread_mutex_unlock(&ed->mbs_lock);
        }

        {
            pthread_mutex_lock(&h->lock[WRITE]);
            while (h->write_cnt>= DPB_SIZE)
                pthread_cond_wait(&h->cond[WRITE], &h->lock[WRITE]);
            assert(s);
            assert(s->current_picture);
            h->write_q[h->write_fi]= s->current_picture;
            h->write_cnt++;
            h->write_fi++; h->write_fi %= DPB_SIZE;
            pthread_cond_signal(&h->cond[WRITE]);
            pthread_mutex_unlock(&h->lock[WRITE]);

        }
        {
            atomic_dec(rl_cnt);
        }

    }

    {//propagate exit
        pthread_mutex_lock(&h->lock[WRITE]);
        while (h->write_cnt>= DPB_SIZE)
            pthread_cond_wait(&h->cond[WRITE], &h->lock[WRITE]);
        last_pic.reference = -1;
        h->write_q[h->write_fi] = &last_pic;
        h->write_cnt++;
        h->write_fi++; h->write_fi %= DPB_SIZE;
        pthread_cond_signal(&h->cond[WRITE]);
        pthread_mutex_unlock(&h->lock[WRITE]);

    }
    pthread_join(h->rl_dist_thr, NULL);
    pthread_exit(NULL);
    return NULL;
}

/*
* The following code is the main loop of the file converter
*/
int h264_decode_cell(H264Context *h) {

    pthread_t read_thr, parsenal_thr, entropy_thr, mbdec_thr, write_thr;

    start_timer();

    pthread_create(&read_thr, NULL, read_thread, h);
    pthread_create(&parsenal_thr, NULL, parsenal_thread, h);
    pthread_create(&entropy_thr, NULL, entropy_IPB_cell_thread, h);
    pthread_create(&mbdec_thr, NULL, mbdec_cell_thread, h);
    pthread_create(&write_thr, NULL, write_thread, h);

    pthread_join(read_thr, NULL);
    pthread_join(parsenal_thr, NULL);
    pthread_join(entropy_thr, NULL);
    pthread_join(mbdec_thr, NULL);
    pthread_join(write_thr, NULL);

    return 0;
}

/*
* The following code is the main loop of the file converter
*/
int h264_decode_cell_seq(H264Context *h) {
ParserContext *pc;
    NalContext *nc;
    EntropyContext *ec;
    MBRecContext *rc;
    OutputContext *oc;

    RawFrame frm;
    EDSlice slice, *s=&slice;
    MBSlice mbslice, *s2=&mbslice;
    PictureInfo *pic=NULL;
    DecodedPicture *out;
    int size;
    int frames=0;

    pc = get_parse_context(h->ifile);
    nc = get_nal_context(h->width, h->height);
    ec = get_entropy_context( h );
    rc = get_mbrec_context(h);
    oc = get_output_context( h );

    rl_lock = (mutex_ea_t) (unsigned) &rl_mutex_var;
    rl_cond = (cond_ea_t) (unsigned) &rl_cond_var;
    rl_cnt = (atomic_ea_t) (unsigned) &rl_cnt_var;
    atomic_set(rl_cnt, 0);
    mutex_init(rl_lock);
    cond_init(rl_cond);

    memset(s, 0, sizeof(EDSlice));
    ff_init_slice(nc, s);
    s->mbs = av_malloc( h->mb_height * h->mb_width * sizeof(H264Mb));

    DecodedPicture tmp;
    tmp.base[0]=0;
    ///fix this when want to debug the Cell errors
    //init_dpb_entry(&tmp, h->width, h->height);

    create_spe_ED_threads(h, 1, 0);
    create_spe_MBR_threads(h, 1);

    start_timer();

    while(!pc->final_frame && frames++ < h->num_frames){

        av_read_frame_internal(pc, &frm);

        PictureInfo *pic=get_pib_entry(h);
        ff_alloc_picture_info(nc, s, pic);
        decode_nal_units(nc, s, &frm);

        copyEDtoMBSlice(s2, s);
        decode_slice_entropy_cell_seq(h, ec, s);

        decode_slice_mb_seq_cell(h, rc, s2, &tmp);

        out =output_frame(h, oc, s2->current_picture, h->ofile, h->frame_width, h->frame_height);

        if (out){
            release_dpb_entry(h, out, 1);
        }
        print_report(oc->frame_number, oc->video_size, 0, h->verbose);
    }
    while ((out=output_frame(h, oc, NULL, h->ofile, h->frame_width, h->frame_height))) ;

    print_report(oc->frame_number, oc->video_size, 1, h->verbose);

    /* finished ! */
    av_freep(&s->mbs);

    free_parse_context(pc);
    free_nal_context  (nc);
    free_entropy_context(ec);
    free_mbrec_context(rc);
    free_output_context(oc);
    return 0;
}
