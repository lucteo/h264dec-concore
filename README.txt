App: h264dec

This application decodes H.264 raw videos.

Build Sequential/Pthreads:

autoreconf -i -f
mkdir build
cd build
../configure --enable-ssse3 --enable-sdl2
make

Build OmpSs:

autoreconf -i -f
mkdir build
cd build-ss
../configure CC=sscc --enable-ssse3 --enable-sdl2
make

ssse3 enables assembler optimizations up to ssse3 (optional)
sdl enables a rudimentary viewing capability      (optional)

Usage Sequential/Pthreads:
./h264dec -i $(INPUT_VIDEO) -s
./h264dec -i $(INPUT_VIDEO) -t $(THREADS)

Usage OmpSs:
NX_PES=<numthreads> ./h264dec -i <inputfile> -e <num parallel entropy frames> -z <width> <height> --static-3d

-e specify the number of entropy decode pipeline buffers and should be ideally
the same as the number of threads.

-z allows to set the MB reconstruction grouped block size. A size between 6 by 6 to 10 by 10
was found to strike a good balance between overhead and parallelism, but is machine and input
dependent.

--static-3d performs overlapping wavefront decoding.

General usage:
-d 				displays output
-f 				fullscreen
-o $(OUT_FILE)  write raw YUV
-v  			show framerate


The INPUT_VIDEOs are in "inputs_encore", but should be able to decode any raw H.264 stream using
one slice per frame, non-interlaced, and CABAC, YUV420.


Integrated OmpSs player demo
----------------------------
NOTE: for the player demo SDL2 must be installed.

1. Go to the OmpSs build directory (/home/cchi/Projects/ffmpeg_smp/build-ss)

2. Launch the H.264 decoder with the desired options:

NX_PES=<numthreads> ./h264dec <inputfile> -v (verbose) -e <num parallel entropy frames> -z <width> <height> -d (display) -f (fullscreen)

note that <num parallel entropy frames> should be equal or higher than <numthreads> for optimal performance

Examples:

NX_PES=7 ./h264dec -i ../../h264_movies/park_joy_2160px5.h264 -v -z 8 8 -df -e 9
NX_PES=7 ./h264dec -i ../../h264_movies/big_buck_bunny_1080p24.h264 -v -d  -z 6 6 -e 9

Interacting with the program
----------------------------
<CTRL+F>    Fullscreen mode
<ESCAPE>    Window mode
<SPACE>     Pause/resume
<M>         Show/hide macroblock borders
<arrows>    When macroblock borders are shown resizes the macroblocks
<ALT+F4>    Close

Force close in case of lockup
-----------------------------
On a terminal: killall -9 h264dec
