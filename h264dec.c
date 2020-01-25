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
 * H264 decoder main
 * @author Chi Ching Chi <chi.c.chi@tu-berlin.de>
 */

#include "config.h"
#include "libavcodec/h264.h"

#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>

#include <assert.h>


static const char program_name[] = "h264dec";
static const int program_birth_year = 2010;

static const char *file_name;
static int ifile, ofile;
static int no_arch =0;
static int parallel = 1;
static int frame_width  = 0;
static int frame_height = 0;

static void av_exit(int ret)
{
    //do some free calls
#undef exit
    exit(ret);
}

static void opt_input_file(const char *filename)
{
    /* open the input file */
    ifile = open(filename, O_RDONLY, 0666);
    if (ifile < 0){
        fprintf(stderr, "Failed to open %s\n", filename);
        av_exit(-1);
    }

    //parse first frame to get resolution (other information available but not used)
    H264Slice slice;
    PictureInfo pi;
    GetBitContext gb = {0,};
    ParserContext *pc;
    NalContext *nc;

    pc = get_parse_context(ifile);
    nc = get_nal_context(0, 0);

    memset(&slice, 0, sizeof(H264Slice));
    slice.current_picture_info=&pi;

    av_read_frame_internal(pc, &gb);
    decode_nal_units(nc, &slice, &gb);

    frame_width = nc->width;
    frame_height= nc->height;

    //clean up
    av_freep(&gb.raw);
    if (gb.rbsp)
        av_freep(&gb.rbsp);
    free_parse_context(pc);
    free_nal_context(nc);

    //rewind file
    int offset;
    if ( (offset=lseek(ifile, 0, SEEK_SET)) ){
        fprintf(stderr, "Rewind input file %s failed at offset %d\n", filename, offset);
    }

}

static void opt_output_file(const char *filename)
{
    if (filename){
        if (!strcmp(filename, "-"))
            filename = "pipe:";

        ofile = open(filename, O_CREAT | O_TRUNC | O_WRONLY, 0666);
    }else{
        ofile =0;
    }
}

static void show_usage(void)
{
    fprintf(stderr, "usage: ffmpeg -i infile [optional]}\n"
      "Optional arguments:\n"
#if SDL2
      "-d    : display video using SDL2\n"
      "-f    : fullscreen\n"
#endif
#if OMPSS
      "-z <width height>: wavefront block size\n"
      "-e <num>         : pipeline buffers\n"
      "-w               : add mbd tasks in wavefront order instead of raster scan\n"
      "--static-3d      : use overlapping wavefront execution\n"
      "--framerate  <float> : target frames per second\n"
      "--framedelay <float> : maximum framedelay before a frame is considered late\n"
#else
      "-t <num>   : number of threads\n"
      "-p <0|1> : pin threads\n"
      "--smt    : use hardcoded SMT core number table for pinning\n"
      "--slice-bufs <num> : number of slice buffers\n"
#endif
      "-s       : non-threaded execution\n"
      "-o       : raw output file\n"
      "-n <num> : number of frames\n"
      "-v       : verbose progress report\n"
      "-q       : profile stages\n"
      "--no-mbd : skip macroblock reconstruction\n"
      "\n"
    );
}

static struct option long_options[] = {
    {"static-mbd", 0, 0, 0},
    {"numamap", 0, 0, 0},
    {"no-mbd", 0, 0, 0},
    {"static-3d", 0, 0, 0},
    {"ompss-rt", 0, 0, 0},
    {"slice-bufs", 1, 0, 0},
    {"smt", 0, 0, 0},
    {"framerate", 1, 0, 0},
    {"framedelay", 1, 0, 0},
    {"noarch", 0, 0, 'a'},
    {"display", 0, 0, 'd'},
    {"fullscreen", 0, 0, 'f'},
    {"numframes", 1, 0, 'n'},
    {"use-ppe-ed", 1, 0, 'p'},
    {"sequential", 0, 0, 's'},
    {"threads", 1, 0, 't'},
    {"verbose", 1, 0, 'v'},
    {"wave-order", 1, 0, 'w'},
    {"smb-size", 1, 0, 'z'},
    {"pipe-bufs", 1, 0, 'e'},
    {0, 0, 0, 0}
};

static h264_options cli_opts;
static void parse_cmd(int argc, char **argv)
{
    int c;
    int digit_optind = 0;
    int option_index = 0;
    char ofile_name[1024];
    extern char *optarg;
    extern int optind, optopt;

    cli_opts.statsched =0;
    cli_opts.numamap =0;
    cli_opts.statmbd =0;
    cli_opts.no_mbd= 0;
    cli_opts.numframes = INT_MAX;
    cli_opts.display=0;
    cli_opts.fullscreen=0;
    cli_opts.verbose=0;
    cli_opts.ppe_ed=0;
    cli_opts.profile=0;
    cli_opts.threads = 1;
    cli_opts.smb_size[0] = cli_opts.smb_size[1] = 1;
    cli_opts.wave_order=0;
    cli_opts.static_3d=0;
    cli_opts.pipe_bufs=8;
    cli_opts.slice_bufs=1;
    cli_opts.smt= 0;
    cli_opts.framerate= 0;
    cli_opts.framedelay= 0.1;
    while ((c = getopt_long(argc, argv, "ade:fi:n:o:p:st:vwz:", long_options, &option_index)) != -1 ){
        int this_option_optind = optind ? optind : 1;

        switch (c){
            case 0:
                if (option_index==0){
                    cli_opts.statmbd= 1;
                }else if (option_index==1){
                    cli_opts.numamap= 1;
                }else if (option_index==2){
                    cli_opts.no_mbd= 1;
                }else if (option_index==3){
                    cli_opts.static_3d= 1;
		}else if (option_index==4){
                    cli_opts.ompss_rt= 1;
                }else if (option_index==5){
                    cli_opts.slice_bufs= (unsigned) atoi(optarg);
                }else if (option_index==6){
                    cli_opts.smt= 1;
                }else if (option_index==7){
                    cli_opts.framerate= atof(optarg);
                }else if (option_index==8){
                    cli_opts.framedelay= atof(optarg);
                }
                break;
            case '0':
            case '1':
            case '2':
                if (digit_optind != 0 && digit_optind != this_option_optind)
                    printf("digits occur in two different argv-elements.\n");
                digit_optind = this_option_optind;
                printf("option %c\n", c);
                break;
            case 'a':
                no_arch=1;
                break;
            case 'd':
                cli_opts.display=1;
                break;
            case 'f':
                cli_opts.fullscreen=1;
                break;
            case 'i':
                file_name = (const char *)optarg;
                opt_input_file(file_name);
                break;
            case 'n':
                cli_opts.numframes = (unsigned) atoi(optarg);
                break;
            case 'o':
                strcpy(ofile_name, optarg);
                opt_output_file(ofile_name);
                break;
            case 'p':
                cli_opts.statsched = atoi(optarg)!=0;
                break;
            case 'q':
                cli_opts.profile = (unsigned) atoi(optarg);
                break;
            case 's':
                cli_opts.threads = 0;
                parallel = 0;
                break;
            case 't':
                cli_opts.threads = atoi(optarg);
                if (cli_opts.threads<=0){
                    fprintf(stderr, "Option -%c requires thread numbers > 0\n", c);
                    av_exit(-1);
                }
                break;
            case 'v':
                cli_opts.verbose = 1;
                break;
            case 'w':
                cli_opts.wave_order = 1;
                break;
            case 'z': // only useful in ompss
                if (argc < optind +1){
                    fprintf(stderr, "Option -%c (--smb-size) requires 2 arguments\n", c);
                    av_exit(-1);
                }
                optind--;
                for (int i=0; i<2; i++){
                    cli_opts.smb_size[i] = atoi(argv[optind++]);
                    if (!(cli_opts.smb_size > 0)){
                        fprintf(stderr, "Option -%c (--smb-size) requires dimensions > 0\n", c);
                        av_exit(-1);
                    }
                }
                break;
            case 'e':
                cli_opts.pipe_bufs = atoi(optarg);
                break;
            case ':':
                fprintf(stderr, "Option -%c requires an operand\n", optopt);
                av_exit(-1);
                break;
            case '?':
                fprintf(stderr, "Unrecognized option: -%c\n", optopt);
                av_exit(-1);
                break;
        }
    }

}

int main(int argc, char **argv)
{
    /* parse options */
    parse_cmd(argc, argv);

    if( !ifile ) {
        show_usage();
        av_exit(1);
    }

    H264Context *h = get_h264dec_context(file_name, ifile, ofile, frame_width, frame_height, &cli_opts);
#if OMPSS
    if (h264_decode_ompss( h ) < 0)
        av_exit(-1);
#else
    if (parallel){
        if (ARCH_CELL && !no_arch){
            if (h264_decode_cell( h ) < 0)
                av_exit(-1);
        }else{
            if (h264_decode_pthread( h ) < 0)
                av_exit(1);
        }
    }else{
        if (ARCH_CELL && !no_arch){
            if (h264_decode_cell_seq( h ) < 0)
                av_exit(1);
        }else{
            if (h264_decode_seq( h ) < 0)
                av_exit(1);
        }
    }
#endif
    free_h264dec_context(h);
    close(ifile);
    close(ofile);

    return 0;
}
