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
#include "h264_data.h"

#include "golomb.h"
#include "h264_sei.h"
#include "h264_refs.h"
#include "h264_ps.h"
#include "h264_pred_mode.h"
#include "h264_misc.h"

static int ff_h264_decode_rbsp_trailing(const uint8_t *src){
    int v= *src;
    int r;

    for(r=1; r<9; r++){
        if(v&1) return r;
        v>>=1;
    }
    return 0;
}

static int pred_weight_table(H264Slice *s, GetBitContext *gb){
    int luma_def, chroma_def;

    s->use_weight= 0;
    s->use_weight_chroma= 0;
    s->luma_log2_weight_denom= get_ue_golomb(gb);
    s->chroma_log2_weight_denom= get_ue_golomb(gb);
    luma_def = 1<<s->luma_log2_weight_denom;
    chroma_def = 1<<s->chroma_log2_weight_denom;

    for(int list=0; list<2; list++){
        for(int i=0; i<s->ref_count[list]; i++){
            int luma_weight_flag, chroma_weight_flag;

            luma_weight_flag= get_bits1(gb);
            if(luma_weight_flag){
                s->luma_weight[i][list][0]= get_se_golomb(gb);
                s->luma_weight[i][list][1]= get_se_golomb(gb);
                if(   s->luma_weight[i][list][0] != luma_def
                    || s->luma_weight[i][list][1] != 0) {
                    s->use_weight= 1;
                }
            }else{
                s->luma_weight[i][list][0]= luma_def;
                s->luma_weight[i][list][1]= 0;
            }

            chroma_weight_flag= get_bits1(gb);
            if(chroma_weight_flag){
                int j;
                for(j=0; j<2; j++){
                    s->chroma_weight[i][list][j][0]= get_se_golomb(gb);
                    s->chroma_weight[i][list][j][1]= get_se_golomb(gb);
                    if(   s->chroma_weight[i][list][j][0] != chroma_def
                    || s->chroma_weight[i][list][j][1] != 0) {
                        s->use_weight_chroma= 1;
                    }
                }
            }else{
                int j;
                for(j=0; j<2; j++){
                    s->chroma_weight[i][list][j][0]= chroma_def;
                    s->chroma_weight[i][list][j][1]= 0;
                }
            }
        }
        if(s->slice_type_nos != FF_B_TYPE) break;
    }
    s->use_weight= s->use_weight || s->use_weight_chroma;
    return 0;
}

/**
* Initialize implicit_weight table.
*/
static void implicit_weight_table(H264Slice *s){
    int ref0, ref1, cur_poc, ref_start, ref_count0, ref_count1;

    cur_poc = s->poc;
    if(   s->ref_count[0] == 1 && s->ref_count[1] == 1  && s->ref_list[0][0]->poc + s->ref_list[1][0]->poc == 2*cur_poc){
        s->use_weight= 0;
        s->use_weight_chroma= 0;
        return;
    }
    ref_start= 0;
    ref_count0= s->ref_count[0];
    ref_count1= s->ref_count[1];

    s->use_weight= 2;
    s->use_weight_chroma= 2;
    s->luma_log2_weight_denom= 5;
    s->chroma_log2_weight_denom= 5;

    for(ref0=ref_start; ref0 < ref_count0; ref0++){
        int poc0 = s->ref_list[0][ref0]->poc;
        for(ref1=ref_start; ref1 < ref_count1; ref1++){
            int poc1 = s->ref_list[1][ref1]->poc;
            int td = av_clip(poc1 - poc0, -128, 127);
            int w= 32;
            if(td){
                int tb = av_clip(cur_poc - poc0, -128, 127);
                int tx = (16384 + (FFABS(td) >> 1)) / td;
                int dist_scale_factor = (tb*tx + 32) >> 8;
                if(dist_scale_factor >= -64 && dist_scale_factor <= 128)
                    w = 64 - dist_scale_factor;
            }
            s->implicit_weight[ref0][ref1][0]=
            s->implicit_weight[ref0][ref1][1]= w;
        }
    }
}

/**
* instantaneous decoder refresh.
*/
static void idr(NalContext *n, H264Slice *s){
    ff_h264_remove_all_refs(n, s);
    n->prev_frame_num= 0;
    n->prev_frame_num_offset= 0;
    n->poc_offset +=  (n->prev_poc_msb<<16) + n->prev_poc_lsb;
    n->prev_poc_msb=
    n->prev_poc_lsb= 0;
}

static int init_poc(NalContext *n, H264Slice *s, GetBitContext *gb){
    const int max_frame_num= 1<<n->sps.log2_max_frame_num;
    int frame_poc;

    if(n->sps.poc_type==0){
        n->poc_lsb= get_bits(gb, n->sps.log2_max_poc_lsb);
    }

    if(n->sps.poc_type==1 && !n->sps.delta_pic_order_always_zero_flag){
        n->delta_poc= get_se_golomb(gb);
    }

    n->frame_num_offset= n->prev_frame_num_offset;
    if(n->frame_num < n->prev_frame_num)
        n->frame_num_offset += max_frame_num;

    if(n->sps.poc_type==0){
        const int max_poc_lsb= 1<<n->sps.log2_max_poc_lsb;

        if(n->poc_lsb < n->prev_poc_lsb && n->prev_poc_lsb - n->poc_lsb >= max_poc_lsb/2)
            n->poc_msb = n->prev_poc_msb + max_poc_lsb;
        else if(n->poc_lsb > n->prev_poc_lsb && n->prev_poc_lsb - n->poc_lsb < -max_poc_lsb/2)
            n->poc_msb = n->prev_poc_msb - max_poc_lsb;
        else
            n->poc_msb = n->prev_poc_msb;

        frame_poc = n->poc_msb + n->poc_lsb;
    }else if(n->sps.poc_type==1){
        int abs_frame_num, expected_delta_per_poc_cycle, expectedpoc;
        int i;

        if(n->sps.poc_cycle_length != 0)
            abs_frame_num = n->frame_num_offset + n->frame_num;
        else
            abs_frame_num = 0;

        if(s->nal_ref_idc==0 && abs_frame_num > 0)
            abs_frame_num--;

        expected_delta_per_poc_cycle = 0;
        for(i=0; i < n->sps.poc_cycle_length; i++)
            expected_delta_per_poc_cycle += n->sps.offset_for_ref_frame[ i ]; //FIXME integrate during sps parse

        if(abs_frame_num > 0){
            int poc_cycle_cnt          = (abs_frame_num - 1) / n->sps.poc_cycle_length;
            int frame_num_in_poc_cycle = (abs_frame_num - 1) % n->sps.poc_cycle_length;

            expectedpoc = poc_cycle_cnt * expected_delta_per_poc_cycle;
            for(i = 0; i <= frame_num_in_poc_cycle; i++)
                expectedpoc = expectedpoc + n->sps.offset_for_ref_frame[ i ];
        } else
            expectedpoc = 0;
        if(s->nal_ref_idc == 0)
            expectedpoc = expectedpoc + n->sps.offset_for_non_ref_pic;
        frame_poc = expectedpoc + n->delta_poc;
    }else{
        int poc= 2*(n->frame_num_offset + n->frame_num);
        if(!s->nal_ref_idc)
            poc--;
        frame_poc= poc;
    }
    s->current_picture_info->poc= s->poc = frame_poc + n->poc_offset;
    s->coded_pic_num = n->coded_pic_num++;

    return 0;
}

static void ref2frame(NalContext *n, H264Slice *s){
    for(int j=0; j<s->list_count; j++){
        int *ref2frm= s->ref2frm[j];

        ref2frm[0]=
        ref2frm[1]= -1;

        for(int i=0; i<s->ref_count[j]; i++){
            ref2frm[i+2]= 15;
            if(s->ref_list[j][i]->cpn >=0){
                int k;
                for(k=0; k<n->short_ref_count; k++){
                    if(n->short_ref[k]->cpn == s->ref_list[j][i]->cpn){
                        ref2frm[i+2]= k;
                        break;
                    }
                }
            }
        }
    }
}

/**
* decodes a slice header.
* This will also call MPV_common_init() and frame_start() as needed.
*
* @param h h264context
* @param h0 h264 master context (differs from 'h' when doing sliced based parallel decoding)
*
* @return 0 if okay, <0 if an error occurred, 1 if decoding must not be multithreaded
*/
static int decode_slice_header(NalContext *n, H264Slice *s, GetBitContext *gb){
    unsigned int first_mb_in_slice;
    unsigned int pps_id;
    int num_ref_idx_active_override_flag;
    unsigned int slice_type, tmp;

    first_mb_in_slice= get_ue_golomb(gb);
    (void) first_mb_in_slice;

    slice_type= get_ue_golomb_31(gb);
    if(slice_type > 9){
        av_log(AV_LOG_ERROR, "slice type too large (%d)\n", s->slice_type);
        return -1;
    }
    if(slice_type > 4)
        slice_type -= 5;

    slice_type= golomb_to_pict_type[ slice_type ];

    s->slice_type= slice_type;
    s->slice_type_nos= slice_type & 3;
    s->current_picture_info->slice_type_nos = s->slice_type_nos;
    s->current_picture_info->reference= s->nal_ref_idc? 2:0;
    s->key_frame = s->slice_type == FF_I_TYPE;

    pps_id= get_ue_golomb(gb);

    if(pps_id>=MAX_PPS_COUNT){
        av_log(AV_LOG_ERROR, "pps_id out of range\n");
        return -1;
    }
    if(!n->pps_buffers[pps_id]) {
        av_log(AV_LOG_ERROR, "non-existing PPS %u referenced\n", pps_id);
        return -1;
    }
    s->pps= *n->pps_buffers[pps_id];

    if(!n->sps_buffers[s->pps.sps_id]) {
        av_log(AV_LOG_ERROR, "non-existing SPS %u referenced\n", s->pps.sps_id);
        return -1;
    }
    n->sps = *n->sps_buffers[s->pps.sps_id];

    n->mb_width= n->sps.mb_width;
    n->mb_height= n->sps.mb_height;

    int chroma444 = (n->sps.chroma_format_idc == 3);
    n->width = 16*n->mb_width - (2>>chroma444)*FFMIN(n->sps.crop_right, (8<<chroma444)-1);
    if(n->sps.frame_mbs_only_flag)
        n->height= 16*n->mb_height - (2>>chroma444)*FFMIN(n->sps.crop_bottom, (8<<chroma444)-1);
    else
        n->height= 16*n->mb_height - (4>>chroma444)*FFMIN(n->sps.crop_bottom, (8<<chroma444)-1);

    s->direct_8x8_inference_flag = n->sps.direct_8x8_inference_flag;
    s->transform_bypass = n->sps.transform_bypass;

    n->frame_num= get_bits(gb, n->sps.log2_max_frame_num);
    if(n->frame_num !=  n->prev_frame_num && n->frame_num != (n->prev_frame_num+1)%(1<<n->sps.log2_max_frame_num)){
        av_log(AV_LOG_ERROR, "unexpected frame_num \n");
    }

    s->current_picture_info->frame_num= n->frame_num; //FIXME frame_num cleanup
    n->max_pic_num= 1<< n->sps.log2_max_frame_num;

    if(s->nal_unit_type == NAL_IDR_SLICE){
        get_ue_golomb(gb); /* idr_pic_id */
    }

    init_poc(n, s, gb);

    if(s->pps.redundant_pic_cnt_present){
        n->redundant_pic_count= get_ue_golomb(gb);
    }

    //set defaults, might be overridden a few lines later
    s->ref_count[0]= s->pps.ref_count[0];
    s->ref_count[1]= s->pps.ref_count[1];

    if(s->slice_type_nos != FF_I_TYPE){
        if(s->slice_type_nos == FF_B_TYPE){
            s->direct_spatial_mv_pred= get_bits1(gb);
        }
        num_ref_idx_active_override_flag= get_bits1(gb);

        if(num_ref_idx_active_override_flag){
            s->ref_count[0]= get_ue_golomb(gb) + 1;
            if(s->slice_type_nos==FF_B_TYPE)
                s->ref_count[1]= get_ue_golomb(gb) + 1;

            if(s->ref_count[0]-1 > 32-1 || s->ref_count[1]-1 > 32-1){
                av_log(AV_LOG_ERROR, "reference overflow\n");
                s->ref_count[0]= s->ref_count[1]= 1;
                return -1;
            }
        }
        if(s->slice_type_nos == FF_B_TYPE)
            s->list_count= 2;
        else
            s->list_count= 1;
    }else
        s->list_count= 0;


    if(s->slice_type_nos!=FF_I_TYPE){
        ff_h264_fill_default_ref_list(n, s);
        ff_h264_decode_ref_pic_list_reordering(n, s, gb);
        ref2frame(n, s);

        for(int i=0; i<2; i++){
            for(int j=0; j<s->ref_count[i]; j++){
                if (s->ref_list[i][j]==NULL || s->ref_list[i][j]->reference < 2) // Don't know why sometimes the ref_count=1 while there are no references
                    s->ref_list_cpn[i][j] = -1;
                else
                    s->ref_list_cpn[i][j] = s->ref_list[i][j]->cpn;
            }
        }
    }

    if(   (s->pps.weighted_pred          && s->slice_type_nos == FF_P_TYPE )
    ||  (s->pps.weighted_bipred_idc==1 && s->slice_type_nos== FF_B_TYPE ) ){
        pred_weight_table(s, gb);
    }
    else if(s->pps.weighted_bipred_idc==2 && s->slice_type_nos== FF_B_TYPE){
        implicit_weight_table( s);
    }else {
        s->use_weight = 0;
    }

    if(s->nal_ref_idc){
        ff_h264_ref_pic_marking(n, s, gb);
        n->prev_poc_msb= n->poc_msb;
        n->prev_poc_lsb= n->poc_lsb;
    }

    n->prev_frame_num_offset= n->frame_num_offset;
    n->prev_frame_num= n->frame_num;

    if(s->slice_type_nos != FF_B_TYPE){
        s->ip_id= n->ip_id++;
    }

    if(s->slice_type_nos==FF_B_TYPE && !s->direct_spatial_mv_pred){
        ff_h264_direct_dist_scale_factor(s);
    }
    ff_h264_direct_ref_list_init(s);


    if( s->slice_type_nos != FF_I_TYPE && s->pps.cabac ){
        tmp = get_ue_golomb_31(gb);
        if(tmp > 2){
            av_log(AV_LOG_ERROR, "cabac_init_idc overflow\n");
            return -1;
        }
        s->cabac_init_idc= tmp;
    }

    tmp = s->pps.init_qp + get_se_golomb(gb);
    if(tmp>51){
        av_log(AV_LOG_ERROR, "QP %u out of range\n", tmp);
        return -1;
    }
    s->qscale= tmp;

    //FIXME qscale / qp ... stuff
    if(s->slice_type == FF_SP_TYPE){
        get_bits1(gb); /* sp_for_switch_flag */
    }
    if(s->slice_type==FF_SP_TYPE || s->slice_type == FF_SI_TYPE){
        get_se_golomb(gb); /* slice_qs_delta */
    }

    s->slice_alpha_c0_offset = 52;
    s->slice_beta_offset = 52;
    if( s->pps.deblocking_filter_parameters_present ) {
        tmp= get_ue_golomb_31(gb);
        if(tmp > 1){
            av_log(AV_LOG_ERROR, "deblocking_filter_idc %u out of range\n", tmp);
            return -1;
        }

        if(tmp < 2)
            tmp^= 1; // 1<->0

        if( tmp ) {
            s->slice_alpha_c0_offset += get_se_golomb(gb) << 1;
            s->slice_beta_offset     += get_se_golomb(gb) << 1;
            if( (unsigned) s->slice_alpha_c0_offset > 104U
            ||(unsigned) s->slice_beta_offset    > 104U){
                av_log(AV_LOG_ERROR, "deblocking filter parameters %d %d out of range\n", s->slice_alpha_c0_offset, s->slice_beta_offset);
                return -1;
            }
        }
    }

    s->qp_thresh= 15 + 52 - FFMIN(s->slice_alpha_c0_offset, s->slice_beta_offset) - FFMAX3(0, s->pps.chroma_qp_index_offset[0], s->pps.chroma_qp_index_offset[1]);

    return 0;
}

PictureInfo *get_pib_entry(NalContext *nc, int coded_pic_num){
    PictureInfo *pic = NULL;

    for(int i=0; i<MAX_REF_PIC_COUNT+1; i++){
        if(nc->picture[i].reference==0){
            pic= &nc->picture[i];
            break;
        }
    }
    pic->cpn = coded_pic_num;

    return pic;
}

int decode_nal_units(NalContext *n, H264Slice *s, GetBitContext *gb1){
    GetBitContext *gb = gb1;
    uint8_t *buf = gb1->raw;
    int buf_size = gb1->buf_size;
    int next_avc = buf_size;
    int buf_index=0;
    uint8_t *dst=NULL;
//     gb->raw = gb1->raw;
//     gb->rbsp = NULL;
    s->release_cnt=0;
    ff_h264_reset_sei(n);

    s->current_picture_info = get_pib_entry(n, n->coded_pic_num);

    for(;;){
        int consumed;
        int dst_length;
        int bit_length;
        const uint8_t *ptr;
        int err;

        if (buf_index >= buf_size){
            break;
        } else {
            // start code prefix search
            for(; buf_index + 3 < buf_size; buf_index++){
                // This should always succeed in the first iteration.
                if(buf[buf_index] == 0 && buf[buf_index+1] == 0 && buf[buf_index+2] == 1)
                    break;
            }
            if(buf_index+3 >= buf_size) break;
            buf_index+=3;
        }

        {
            int length = next_avc - buf_index;
            int i, si, di;
            uint8_t *src= buf+buf_index;
            //    src[0]&0x80;                //forbidden bit
            s->nal_ref_idc= src[0]>>5;
            s->nal_unit_type= src[0]&0x1F;

            src++; length--;

            for(i=0; i+1<length; i+=2){
                if(src[i]) continue;
                if(i>0 && src[i-1]==0) i--;
                if(i+2<length && src[i+1]==0 && src[i+2]<=3){
                    if(src[i+2]!=3){
                        /* startcode, so we must be past the end */
                        length=i;
                    }
                    break;
                }
            }

            if(i>=length-1){ //no escaped 0
                dst_length= length;
                consumed= length+1; //+1 for the header
                ptr=src;
            }else{
                av_fast_malloc(&gb->rbsp, &gb->rbsp_size, length+FF_INPUT_BUFFER_PADDING_SIZE);
                dst = gb->rbsp;
//                 if (dst){
//                     av_free(dst);
//                 }
//                 dst = av_malloc(length+FF_INPUT_BUFFER_PADDING_SIZE);

                if (dst == NULL){
                    return -1;
                }

                //printf("decoding esc\n");
                memcpy(dst, src, i);
                si=di=i;
                while(si+2<length){
                    //remove escapes (very rare 1:2^22)
                    if(src[si+2]>3){
                        dst[di++]= src[si++];
                        dst[di++]= src[si++];
                    }else if(src[si]==0 && src[si+1]==0){
                        if(src[si+2]==3){ //escape
                            dst[di++]= 0;
                            dst[di++]= 0;
                            si+=3;
                            continue;
                        }else //next start code
                            goto nsc;
                    }

                    dst[di++]= src[si++];
                }
                while(si<length)
                    dst[di++]= src[si++];
                nsc:

                memset(dst+di, 0, FF_INPUT_BUFFER_PADDING_SIZE);

                dst_length= di;
                consumed= si + 1;//+1 for the header
                //FIXME store exact number of bits in the getbitcontext (it is needed for decoding)
                ptr=dst;
//                 gb->rbsp=ptr;
            }
        }
        if (ptr==NULL || dst_length < 0){
            return -1;
        }

        //error prevention, should not touch dst_length
        while(ptr[dst_length - 1] == 0 && dst_length > 0)
            dst_length--;

        bit_length= !dst_length ? 0 : (8*dst_length - ff_h264_decode_rbsp_trailing(ptr + dst_length - 1));
        buf_index += consumed;

        err = 0;
        init_get_bits(gb, ptr, bit_length);
        switch(s->nal_unit_type){
            case NAL_IDR_SLICE:
                idr(n, s); //FIXME ensure we don't loose some frames if there is reordering
            case NAL_SLICE:
                if((err = decode_slice_header(n, s, gb)))
                    break;
                s->key_frame |= (s->nal_unit_type == NAL_IDR_SLICE) || (n->sei_recovery_frame_cnt >= 0);
                break;
            case NAL_DPA:
            case NAL_DPB:
            case NAL_DPC:
                av_log(AV_LOG_ERROR,"no slices/data partitioning support\n");
                break;
            case NAL_SEI:
                ff_h264_decode_sei(n, gb);
                break;
            case NAL_SPS:
                ff_h264_decode_seq_parameter_set(n, gb);
                break;
            case NAL_PPS:
                ff_h264_decode_picture_parameter_set(n, gb, bit_length);
                break;
            case NAL_AUD:
            case NAL_END_SEQUENCE:
            case NAL_END_STREAM:
            case NAL_FILLER_DATA:
            case NAL_SPS_EXT:
            case NAL_AUXILIARY_SLICE:
                break;
            default:
                av_log(AV_LOG_ERROR, "Unknown NAL code: %d (%d bits)\n", s->nal_unit_type, bit_length);
        }
        if (err < 0)
            av_log(AV_LOG_ERROR, "decode_slice_header error\n");

    }

    return buf_index;
}

NalContext *get_nal_context(int width, int height){
    const int mb_height = (height + 15) / 16;
    const int mb_width  = (width  + 15) / 16;
    const int mb_stride = ((mb_width+1)/16 + 1) *16; //align mb_stride to 16

    NalContext *nc = av_mallocz(sizeof(NalContext));
    nc->width = width;
    nc->height = height;
    nc->mb_height = mb_height;
    nc->mb_width  = mb_width;
    nc->b4_stride = mb_width*4 + 1;
    nc->mb_stride = mb_stride;
    nc->outputed_poc = INT_MIN;

    for(int i=0; i<16; i++){
        nc->picture[i].cpn =-1;
    }

    return nc;
}

void free_nal_context(NalContext *nc){
    for(int i = 0; i < MAX_SPS_COUNT; i++){
        if (nc->sps_buffers[i]){
            av_free( nc->sps_buffers[i]);
        }
    }
    for(int i = 0; i < MAX_PPS_COUNT; i++){
        if (nc->pps_buffers[i]){
            av_free( nc->pps_buffers[i]);
        }
    }
    av_free(nc);
}
