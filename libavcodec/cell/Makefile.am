
AM_CFLAGS = -std=gnu99
bin_PROGRAMS = $(top_builddir)/spe_mbd $(top_builddir)/spe_ed
CC = spu-gcc
LIBS=
__top_builddir__spe_mbd_SOURCES = spe_mbd.c 		\
							h264_decode_mb_spu.c 	\
							h264_intra_spu.c 		\
							dsputil_spu.c 			\
							h264_idct_spu.c			\
							h264_mc_spu.c 			\
							h264_deblock_spu.c		\
							h264_dma.c				\
							h264_tables.c 			

__top_builddir__spe_mbd_LDADD= /opt/cell/sdk/usr/spu/lib/libsync.a
__top_builddir__spe_mbd_CPPFLAGS = -I$(top_srcdir) -I /opt/cell/sdk/usr/spu/include

__top_builddir__spe_ed_SOURCES = spe_ed.c 		\
							h264_cabac_spu.c 	\
							cabac_spu.c 		\
							h264_direct_spu.c   \
							h264_dma.c			
 			
__top_builddir__spe_ed_LDADD= /opt/cell/sdk/usr/spu/lib/libsync.a
__top_builddir__spe_ed_CPPFLAGS = -I$(top_srcdir) -I /opt/cell/sdk/usr/spu/include



