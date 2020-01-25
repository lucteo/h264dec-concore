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

#ifndef H264_CELL_TYPES_H
#define H264_CELL_TYPES_H

#include <libsync.h>
#include <libavcodec/avcodec.h>

typedef struct spe_pos{
	volatile int count;		//number of mb processed
	uint32_t pad[3];
}spe_pos;

//only the picture pointers are needed from the picture struct;
typedef struct Picture_spu {
	uint8_t* data[3];
} Picture_spu;

///For Cell, might be idea to use this instead for everything
// struct that contains the pararms that change on slice
typedef struct H264slice{
	int deblocking_filter;
    int linesize;
    int uvlinesize;
	int mb_width;
	int mb_height;

    int use_weight;
    int use_weight_chroma;
    int luma_log2_weight_denom;
    int chroma_log2_weight_denom;

    int16_t luma_weight[16][2][2];
    int16_t chroma_weight[16][2][2][2];
    int16_t implicit_weight[16][16][2];

	// ref picture ptr
    Picture_spu ref_list[2][16];
	int state;
	int emu_edge_width;
    int emu_edge_height;

    int slice_type;
	int slice_type_nos;
	int slice_alpha_c0_offset;
    int slice_beta_offset;

	uint8_t chroma_qp_table[2][64];

	H264Mb *blocks;
	uint8_t  *dst_y, *dst_cb, *dst_cr;

    //uint32_t pad[2];		// padding the structure for multiple of 16 bytes
}H264slice;

typedef struct 	H264spe{
#define EDIP 0
#define EDB  1
#define MBD  2
	int type;
	int idx;
	int spe_id;
	int spe_total;
	int mb_width;
	int mb_stride;
	int mb_height;
	int linesize;
	int uvlinesize;
	//H264slice* slice_params;
	void* src_spe;
	void* tgt_spe;

	mutex_ea_t lock;
	cond_ea_t cond;
	atomic_ea_t cnt;

	mutex_ea_t rl_lock;
	cond_ea_t rl_cond;
	atomic_ea_t rl_cnt;
}H264spe;

typedef struct H264Cabac_spu{
	int blocking;

    int top_cbp;
    int left_cbp;
    int neighbor_transform_size; //number of neighbors (top and/or left) that used 8x8 dct

    uint32_t dequant4_buffer[6][52][16];
    uint32_t dequant8_buffer[2][52][64];
    uint32_t (*dequant4_coeff[6])[16];
    uint32_t (*dequant8_coeff[2])[64];

    uint8_t (*non_zero_count_top)[32];
	uint8_t (*non_zero_count)[32];

	uint8_t (*mvd_top[2])[2];
	uint8_t (*mvd[2])[2];

	uint8_t *direct_top;
	uint8_t *direct;

	uint8_t *chroma_pred_mode_top;
	uint8_t *chroma_pred_mode;

	int8_t  *intra4x4_pred_mode_top;
    int8_t  *intra4x4_pred_mode;

	uint16_t *cbp_top;
	uint16_t *cbp;

	int8_t *qscale_top;
	int8_t *qscale;

	int8_t *ref_index_top[2];
	int8_t *ref_index[2];

	int16_t (*motion_val_top[2])[2];
	int16_t (*motion_val[2])[2];
	uint32_t *mb_type_top;
	uint32_t *mb_type;

	int8_t *list1_ref_index[2];
	uint32_t *list1_mb_type;
	DECLARE_ALIGNED_16(int16_t, list1_motion_val[2][4*4][2]); // fill for a macroblock when required

	int b_stride;
	int mb_stride;
	int mb_width;
	int mb_height;

    uint8_t zigzag_scan[16];
    uint8_t zigzag_scan8x8[64];

    uint8_t direct_cache[5*8];
    // Used to calculate loopfilter bS.
    DECLARE_ALIGNED(16, int16_t, mv_cache)[2][5*8][2];
    DECLARE_ALIGNED(8, int8_t, ref_cache)[2][5*8];
    DECLARE_ALIGNED(8, uint8_t, non_zero_count_cache)[6*8];
    DECLARE_ALIGNED(16, uint8_t, mvd_cache)[2][5*8][2];

} H264Cabac_spu;

typedef struct EDSlice_spu{
    PPS pps;                 ///< current pps

    H264Mb *mbs;

    int state;
    int qp_thresh;      ///< QP threshold to skip loopfilter

	PictureInfo pic;
	PictureInfo list1;
//    Picture *ref_list[2][16];         ///Reordered version of default_ref_list according to picture reordering in slice header
    int ref_count[2];   ///< counts frames or fields, depending on current mb mode
	int slice_type;
    int slice_type_nos;
	int direct_8x8_inference_flag;

    uint8_t list_count;
    uint32_t coded_pic_num;
///stuff only needed for nal/entropy decoding
    H264Mb *m;
    //GetBitContext gb;
	const uint8_t *bytestream_start;
	int byte_bufsize;
    int transform_bypass;
    int direct_spatial_mv_pred;
    int map_col_to_list0[2][16];
    int dist_scale_factor[16];

    int cabac_init_idc;
    int ref2frm[2][64];  ///< reference to frame number lists, the first 2 are for -2,-1
    int qscale;
    int chroma_qp[2]; //QPc
    int last_qscale_diff;

//  Picture* release_ref[MAX_MMCO_COUNT];
//   int release_cnt;


//     int use_weight;
//     int use_weight_chroma;
//    int luma_log2_weight_denom;
//    int chroma_log2_weight_denom;

//     int8_t luma_weight[16][2][2];
//     int8_t chroma_weight[16][2][2][2];
//     int8_t implicit_weight[16][16][2];



//  int slice_alpha_c0_offset;
//  int slice_beta_offset;

//    int nal_ref_idc;
//    int nal_unit_type;
//     uint8_t *rbsp_buffer;
//     unsigned int rbsp_buffer_size;



} EDSlice_spu;

#endif
