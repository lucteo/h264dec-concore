AM_CFLAGS = -std=gnu99 -Wall #-Werror
AM_LDFLAGS =
SUBDIRS = libavutil libavcodec
bin_PROGRAMS = h264dec
h264dec_SOURCES = h264dec.c

if HAVE_ARCH_SUBDIR
ARCH_LIB=libavcodec/$(ARCH_SUBDIR)/libavcodec_$(ARCH_SUBDIR).a
endif

if HAVE_CELL
#For Cell LD_LIBRARY_PATH not working for some reason
CELL_SYNC_LIB=/opt/cell/sdk/usr/lib/libsync.a
endif

h264dec_LDADD = libavcodec/libavcodec.a $(ARCH_LIB) libavutil/libavutil.a $(CELL_SYNC_LIB)

