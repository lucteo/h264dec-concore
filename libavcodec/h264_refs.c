/*
 * H.26L/H.264/AVC/JVT/14496-10/... reference picture handling
 * Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * H.264 / AVC / MPEG4 part10  reference picture handling.
 * @author Michael Niedermayer <michaelni@gmx.at>
 * @author Chi Ching Chi <chi.c.chi@tu-berlin.de>
 */

#include "dsputil.h"
#include "h264_types.h"
#include "golomb.h"

//#undef NDEBUG
#include <assert.h>

static int build_def_list(PictureInfo **def, PictureInfo **in, int len, int is_long){
    int i[2]={0};
    int index=0;

    while(i[0]<len || i[1]<len){
        while(i[0]<len && !(in[ i[0] ] && (in[ i[0] ]->reference)))
            i[0]++;
        while(i[1]<len && !(in[ i[1] ] && (in[ i[1] ]->reference & 0)))
            i[1]++;
        if(i[0] < len){
            in[ i[0] ]->pic_id= is_long ? i[0] : in[ i[0] ]->frame_num;
            def[index++]= in[ i[0]++ ];
        }
        if(i[1] < len){
            in[ i[1] ]->pic_id= is_long ? i[1] : in[ i[1] ]->frame_num;
            def[index++]= in[ i[1]++ ];
        }
    }

    return index;
}

static int add_sorted(PictureInfo **sorted, PictureInfo **src, int len, int limit, int dir){
    int i, best_poc;
    int out_i= 0;

    for(;;){
        best_poc= dir ? INT_MIN : INT_MAX;

        for(i=0; i<len; i++){
            const int poc= src[i]->poc;
            if(((poc > limit) ^ dir) && ((poc < best_poc) ^ dir)){
                best_poc= poc;
                sorted[out_i]= src[i];
            }
        }
        if(best_poc == (dir ? INT_MIN : INT_MAX))
            break;
        limit= sorted[out_i++]->poc - dir;
    }
    return out_i;
}

int ff_h264_fill_default_ref_list(NalContext *n, H264Slice *s){
    int i,len;

    if(s->slice_type_nos==FF_B_TYPE){
        PictureInfo *sorted[32];
        int cur_poc, list;
        int lens[2];

        cur_poc= s->poc;

        for(list= 0; list<2; list++){
            len= add_sorted(sorted, n->short_ref, n->short_ref_count, cur_poc, !list);
            len+=add_sorted(sorted+len, n->short_ref, n->short_ref_count, cur_poc, list);
            assert(len<=32);
            len= build_def_list(s->ref_list[list], sorted, len, 0);
            len+=build_def_list(s->ref_list[list] +len, n->long_ref, 16 , 1);
            assert(len<=32);

            for(int i=len; i<s->ref_count[list]; i++)
                s->ref_list[list][i] = NULL;

            lens[list]= len;
        }

        if(lens[0] == lens[1] && lens[1] > 1){
            for(i=0; s->ref_list[0][i]->poc == s->ref_list[1][i]->poc && i<lens[0]; i++);

			if(i == lens[0])
				FFSWAP(PictureInfo *, s->ref_list[1][0], s->ref_list[1][1]);
        }
    }else{
        len = build_def_list(s->ref_list[0], n->short_ref, n->short_ref_count, 0);
        len+= build_def_list(s->ref_list[0] +len, n->long_ref, 16, 1);
        assert(len <= 32);
        for(i=len; i<s->ref_count[0]; i++)
            s->ref_list[0][i] = NULL;
    }

    return 0;
}

/**
* print short term list
*/
static void print_short_term(NalContext *n) {
    av_log(AV_LOG_DEBUG, "short term list:\n");
    for(int i=0; i<n->short_ref_count; i++){
        PictureInfo *pic= n->short_ref[i];
        av_log(AV_LOG_DEBUG, "%d fn:%d poc:%d ref:%d \n", i, pic->frame_num, pic->poc, pic->reference);
    }
}

/**
* print long term list
*/
static void print_long_term(NalContext *n) {
    uint32_t i;

    av_log(AV_LOG_DEBUG, "long term list:\n");
    for(i = 0; i < 16; i++){
        PictureInfo *pic= n->long_ref[i];
        if (pic) {
            av_log(AV_LOG_DEBUG, "%d fn:%d poc:%d\n", i, pic->frame_num, pic->poc);
        }
    }
}

int ff_h264_decode_ref_pic_list_reordering(NalContext *n, H264Slice *s, GetBitContext *gb){
    int list, index;

    print_short_term(n);
    print_long_term(n);

    for(list=0; list<s->list_count; list++){

        if(get_bits1(gb)){
            int frame_num = n->frame_num;
            unsigned int abs_diff_pic_num;
            for(index=0; ; index++){
                unsigned int reordering_of_pic_nums_idc= get_ue_golomb_31(gb);
                int i=0;
                PictureInfo *ref = NULL;

                if(reordering_of_pic_nums_idc==3){
                    break;
                }
                if(index >= s->ref_count[list]){
                    av_log(AV_LOG_ERROR, "reference count overflow\n");
                    return -1;
                }

                if (reordering_of_pic_nums_idc>2){
                    av_log(AV_LOG_ERROR, "illegal reordering_of_pic_nums_idc\n");
                    return -1;
                }

                if (reordering_of_pic_nums_idc<2){
                    //av_log(AV_LOG_ERROR, "long term pic not supported\n");

                    abs_diff_pic_num= get_ue_golomb(gb) + 1;
                    if(abs_diff_pic_num > (unsigned) n->max_pic_num){
                        av_log(AV_LOG_ERROR, "abs_diff_pic_num overflow\n");
                        return -1;
                    }

                    if(reordering_of_pic_nums_idc == 0)
                        frame_num-= abs_diff_pic_num;
                    else
                        frame_num+= abs_diff_pic_num;
                    frame_num &= n->max_pic_num - 1;

                    for(i= 0 ; i<n->short_ref_count; i++){
                        ref = n->short_ref[i];
                        if(ref->frame_num == frame_num && ref->reference){
                            break;
                        }
                    }
                    ref->pic_id= frame_num;
                }else{
                    int long_idx;
                    long_idx= get_ue_golomb(gb); //long_term_pic_idx

                    if(long_idx>31){
                        av_log(AV_LOG_ERROR, "long_term_pic_idx overflow\n");
                        return -1;
                    }
                    ref = n->long_ref[long_idx];
                    assert(!(ref && !ref->reference));
                    if(ref && (ref->reference)){
                        ref->pic_id= long_idx;
                        assert(ref->long_ref);
                    }else{
                        av_log(AV_LOG_ERROR, "reference picture missing during reorder\n");
                    }
                }

                if (i >= n->short_ref_count) {
                    av_log(AV_LOG_ERROR, "reference picture missing during reorder\n");
                    return -1;
                } else {
                    for(i=index; i+1 <s->ref_count[list]; i++){

//                         if(ref->frame_num == s->ref_list[list][i]->frame_num)
//                            break;
                        ///there is probably no need for a separate pic_id and frame_num
						if (s->ref_list[list][i]){

							if(ref->long_ref == s->ref_list[list][i]->long_ref && ref->pic_id == s->ref_list[list][i]->pic_id)
								break;
						}
                    }
                    for(; i > index; i--){
                        s->ref_list[list][i]= s->ref_list[list][i-1];
                    }
                    s->ref_list[list][index]= ref;
                }
            }
        }
    }

//     //Check if everything went well
//     for(list=0; list<s->list_count; list++){
// 		//printf("ref_count %d list %d\n", s->ref_count[list], list);
//         for(index= 0; index < s->ref_count[list]; index++){
// 			//printf("%d\n", s->ref_list[list][index]->pic_id);
//             if(!s->ref_list[list][index]->data[0]){
//                 av_log(AV_LOG_ERROR, "Missing reference picture\n");
//                 return -1;
//             }
//         }
//     }

    return 0;
}

static PictureInfo *find_short(NalContext *n, int frame_num){
    int i;
    for(i=0; i<n->short_ref_count; i++){
        if(n->short_ref[i]->frame_num == frame_num) {
            return n->short_ref[i];
        }
    }
    return NULL;
}

static int remove_short(NalContext *n, H264Slice *s, int frame_num, int release){
    int i;

    for (i=0; i<n->short_ref_count; i++){
        if (n->short_ref[i]->frame_num == frame_num){
            if (release){
                s->release_ref_cpn[s->release_cnt++] = n->short_ref[i]->cpn;
                n->short_ref[i]->reference &= ~2;
            }
            n->short_ref[i] = NULL;
            if (--n->short_ref_count)
                memmove(&n->short_ref[i], &n->short_ref[i+1], (n->short_ref_count - i)*sizeof(PictureInfo *));
            return 0;
        }
    }
    return -1;
}

static void remove_long(NalContext *n, H264Slice *s, int i){

    if (n->long_ref[i]){
        s->release_ref_cpn[s->release_cnt++] = n->long_ref[i]->cpn;
        n->long_ref[i]->reference &= ~2;
        n->long_ref[i]->long_ref = 0;
        n->long_ref_count--;
        n->long_ref[i] = NULL;
    }
}

void ff_h264_remove_all_refs(NalContext *n, H264Slice *s){
    int i;

    while (n->short_ref[0])
        remove_short(n, s, n->short_ref[0]->frame_num, 1);

    for(i=0; i<16; i++){
        remove_long(n, s, i);
    }
    assert(n->short_ref_count==0);
    assert(n->long_ref_count==0);
}

int ff_h264_ref_pic_marking(NalContext *n, H264Slice *s, GetBitContext *gb){

    if(s->nal_unit_type == NAL_IDR_SLICE){ //FIXME fields
        get_bits1(gb); //get_bits1(gb) -1; //broken link
        if(get_bits1(gb)){
            av_log(AV_LOG_ERROR, "MMCO_LONG reference management not supported\n");
        }
    }else{
        if(get_bits1(gb)){ // adaptive_ref_pic_marking_mode_flag
            int i,j;
            for(i= 0; i<MAX_MMCO_COUNT; i++) {
                PictureInfo *pic;
                int short_pic_num=0;
                unsigned int long_arg=0;
                MMCOOpcode opcode= get_ue_golomb_31(gb);

                if(opcode==MMCO_SHORT2UNUSED || opcode==MMCO_SHORT2LONG){
                    short_pic_num= (n->frame_num - get_ue_golomb(gb) - 1) & (n->max_pic_num - 1);
                }
                if(opcode==MMCO_SHORT2LONG || opcode==MMCO_LONG2UNUSED || opcode==MMCO_LONG || opcode==MMCO_SET_MAX_LONG){
                    long_arg= get_ue_golomb_31(gb);
                    if(long_arg >= 16){
                        av_log(AV_LOG_ERROR, "illegal long ref in memory management control operation %d\n", opcode);
                        return -1;
                    }
                }

                if(opcode > (unsigned)MMCO_LONG){
                    av_log(AV_LOG_ERROR, "illegal memory management control operation %d\n", opcode);
                    return -1;
                }
                if(opcode == MMCO_END)
                    break;

                switch (opcode){
                    case MMCO_SHORT2UNUSED:
                        remove_short(n, s, short_pic_num, 1);
                        break;
                    case MMCO_SHORT2LONG:
                        pic = find_short(n, short_pic_num);
                        if (n->long_ref[long_arg] != pic)
                            remove_long(n, s, long_arg);
                        remove_short(n, s, short_pic_num, 0);
                        n->long_ref[long_arg]= pic;
                        if (pic){
                            pic->long_ref=1;
                            n->long_ref[long_arg]= pic;
                            n->long_ref_count++;
                        }
                        break;
                    case MMCO_LONG2UNUSED:
                        assert(n->long_ref[long_arg]);
                        remove_long(n, s, long_arg);
                        break;
                    case MMCO_SET_MAX_LONG:
                        for(j=long_arg; j<16; j++)
                            remove_long(n, s, j);
                        break;
                    case MMCO_RESET:
                        while(n->short_ref_count)
                            remove_short(n, s, n->short_ref[0]->frame_num, 1);

                        for(j=0; j < 16; j++)
                            remove_long(n, s, j);

                        s->current_picture_info->poc=
                        s->poc =
                        n->poc_lsb=
                        n->poc_msb=
                        n->frame_num=
                        s->current_picture_info->frame_num= 0;
                        break;
					case MMCO_END:
					case MMCO_LONG:
						break;
                }
            }
        }else{// sliding window ref picture marking
            if(n->short_ref_count == n->sps.ref_frame_count) {
                s->release_ref_cpn[s->release_cnt++] = n->short_ref[n->short_ref_count - 1]->cpn;
                n->short_ref[n->short_ref_count - 1]->reference &= ~2;
                n->short_ref[ n->short_ref_count - 1 ] =NULL;
                n->short_ref_count--;
            }
        }
    }

    if(n->short_ref_count)
        memmove(&n->short_ref[1], &n->short_ref[0], n->short_ref_count*sizeof(PictureInfo *));

    n->short_ref[0]= s->current_picture_info;
    n->short_ref_count++;

    return 0;
}

static int get_scale_factor(H264Slice *s, int poc, int poc1, int i){
    int poc0 = s->ref_list[0][i]->poc;
    int td = av_clip(poc1 - poc0, -128, 127);
    if(td == 0 || s->ref_list[0][i]->long_ref){
        return 256;
    }else{
        int tb = av_clip(poc - poc0, -128, 127);
        int tx = (16384 + (FFABS(td) >> 1)) / td;
        return av_clip((tb*tx + 32) >> 6, -1024, 1023);
    }
}

void ff_h264_direct_dist_scale_factor(H264Slice *s){
    const int poc = s->current_picture_info->poc;
    const int poc1 = s->ref_list[1][0]->poc;

    for(int i=0; i<s->ref_count[0]; i++){
        s->dist_scale_factor[i] = get_scale_factor(s, poc, poc1, i);
    }
}

static void fill_colmap(H264Slice *s, int map[2][16], int list){
    PictureInfo * const ref1 = s->ref_list[1][0];
    int old_ref, rfield;

    /* bogus; fills in for missing frames */
    memset(map[list], 0, sizeof(map[list]));

    for(rfield=0; rfield<2; rfield++){
        for(old_ref=0; old_ref < ref1->ref_count[list]; old_ref++){
            int poc = ref1->ref_poc[list][old_ref];

            for(int j=0; j<s->ref_count[0]; j++){
                if(s->ref_list[0][j]->poc == poc){
                    map[list][old_ref] = j;
                    break;
                }
            }
        }
    }
}

void ff_h264_direct_ref_list_init(H264Slice *s){
    PictureInfo * const cur = s->current_picture_info;
    int list;

    for(list=0; list<2; list++){
        cur->ref_count[list] = s->ref_count[list];
        for(int j=0; j<s->ref_count[list]; j++){
            cur->ref_poc[list][j] = s->ref_list[list][j] ? s->ref_list[list][j]->poc : 0;
        }
    }

    if(s->slice_type_nos != FF_B_TYPE || s->direct_spatial_mv_pred)
        return;

    for(list=0; list<2; list++){
        fill_colmap(s, s->map_col_to_list0, list);
    }
}

