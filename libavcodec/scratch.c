static void *entropy_thread(void *arg){
	H264Context *h = (H264Context *) arg;
	EDSlice *s;
	
	H264Cabac hcabac;
	CABACContext cabac;
	
	ff_init_cabac_states();
	
	if (init_cabac(h, &hcabac)<0)
		return NULL;
	
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
		
		decode_slice_entropy(&hcabac, &cabac, s);
		
		{
			pthread_mutex_lock(&h->lock[MBDEC]);
			while (h->mbdec_cnt >= MAX_SLICE_COUNT)
				pthread_cond_wait(&h->cond[MBDEC], &h->lock[MBDEC]);
			h->mbdec_q[h->mbdec_fi] = *((MBSlice *) s);
			h->mbdec_cnt++;
			h->mbdec_fi++; h->mbdec_fi %= MAX_SLICE_COUNT;
			pthread_cond_signal(&h->cond[MBDEC]);
			pthread_mutex_unlock(&h->lock[MBDEC]);
		}
		{
			pthread_mutex_lock(&h->lock[ENTROPY]);
			h->ed_cnt--;
			pthread_cond_signal(&h->cond[ENTROPY]);
			pthread_mutex_unlock(&h->lock[ENTROPY]);
		}
	}
	
	{
		pthread_mutex_lock(&h->lock[MBDEC]);
		while (h->mbdec_cnt >= MAX_SLICE_COUNT)
			pthread_cond_wait(&h->cond[MBDEC], &h->lock[MBDEC]);
		h->mbdec_q[h->mbdec_fi] = *((MBSlice *) s);
		h->mbdec_cnt++;
		h->mbdec_fi++; h->mbdec_fi %= MAX_SLICE_COUNT;
		pthread_cond_signal(&h->cond[MBDEC]);
		pthread_mutex_unlock(&h->lock[MBDEC]);
		
	}
	
	free_cabac(&hcabac);
	
	pthread_exit(NULL);
	return NULL;
	
}
/*
* The following code is the main loop of the file converter
*/
int av_transcode_1ed(int ifile, int ofile, int frame_width, int frame_height) {
	H264Context *h;
	pthread_t read_thr, parsenal_thr, entropy_thr, mbdec_thr, write_thr;
	
	h = ff_h264_decode_init(ifile, ofile, frame_width, frame_height);
	
	timer_start = av_gettime();
	
	//    pthread_create(&read_thr, NULL, read_thread, h);
	//    pthread_create(&parsenal_thr, NULL, parsenal_thread, h);
	pthread_create(&entropy_thr, NULL, entropy_mbd_thread, h);
	
	// pthread_create(&mbdec_thr, NULL, mbdec_thread, h);
	
	//   pthread_create(&write_thr, NULL, write_thread, h);
	
	//   pthread_join(read_thr, NULL);
	//    pthread_join(parsenal_thr, NULL);
	pthread_join(entropy_thr, NULL);
	//    pthread_join(mbdec_thr, NULL);
	//	printf("before write_thr\n");
	//    pthread_join(write_thr, NULL);
	
	/* finished ! */
	ff_h264_decode_end(h);
	
	return 0;
}

static void reset_h264mb(EDSlice *s, int mb_width, int mb_height){
	for (int i=0; i<mb_height; i++){
		for (int j=0; j<mb_width; j++){
			H264Mb *m = &s->mbs[i*mb_width + j];

			m->left_mb_xy=0;
			m->top_mb_xy = 0;
		}
	}
}

static void *entropy_mbd_thread(void *arg){
	H264Context *h = (H264Context *) arg;

	EDSlice slice, *s=&slice;
	MBSlice mbslice, *s2=&mbslice;
	H264Cabac hcabac;
	CABACContext cabac;
	int frames =0;
	MBDecContext mbdec, *d=&mbdec;
	int size=h->width*h->height;
	WriteContext write, *w=&write;
	AVCodecParserContext parser, *pc= &parser;
	NalContext nal, *n=&nal;


	memset(pc, 0, sizeof(AVCodecParserContext));
	pc->buffer_size = 2048;
	pc->final_frame = 0;
	pc->cur_len= 0;
	pc->data = av_mallocz(2048 + FF_INPUT_BUFFER_PADDING_SIZE);
	pc->size = 2048;
	pc->eof_reached =0;
	pc->ifile = h->ifile;

	//init parse
	memset(n, 0, sizeof(NalContext));
	n->width = h->width;
	n->height = h->height;
	n->mb_height = h->mb_height;
	n->mb_width  = h->mb_width;
	n->b4_stride = n->mb_width*4 + 1;
	n->mb_stride = n->mb_width + 1;
	n->outputed_poc = INT_MIN;
// 	memset(s, 0, sizeof(EDSlice));
// 	ff_init_slice(n, s);
//

	memset(w, 0, sizeof(WriteContext));
	w->bit_buffer_size= FFMAX(1024*256, 6*size + 200);
	w->bit_buffer=  av_mallocz(w->bit_buffer_size);



	ff_h264dsp_init(&d->hdsp);
	ff_h264_pred_init(&d->hpc);
	dsputil_init(&d->dsp);
	d->hdsp.qpel_put= d->dsp.put_h264_qpel_pixels_tab;
	d->hdsp.qpel_avg= d->dsp.avg_h264_qpel_pixels_tab;
	d->mb_height = (h->height + 15) / 16;
	d->mb_width  = (h->width  + 15) / 16;
	d->linesize = h->width + EDGE_WIDTH*2;
	d->uvlinesize = d->linesize>>1;

	for(int i=0; i<16; i++){
		d->block_offset[i]= 4*((scan8[i] - scan8[0])&7) + 4*d->linesize*((scan8[i] - scan8[0])>>3);
	}
	for(int i=0; i<4; i++){
		d->block_offset[16+i]=
		d->block_offset[20+i]= 4*((scan8[i] - scan8[0])&7) + 4*d->uvlinesize*((scan8[i] - scan8[0])>>3);
	}

	d->scratchpad= av_mallocz((h->width+64)*4*16*2*sizeof(uint8_t));

	ff_init_cabac_states();

	if (init_cabac(h, &hcabac)<0)
		return NULL;

	while(!pc->final_frame && frames_max++ < 1000){
		Picture *out;

		RawFrame *frm;
		Picture *pic=NULL;

		RawFrame frm_read;
		frm_read.state =0;
		av_read_frame_internal(pc, &frm_read);
		frm = &frm_read;

		if (frm->state < 0)
			break;
/*
		{
			pthread_mutex_lock(&h->lock[PARSE2]);
			while (h->slice_cnt<=0)
				pthread_cond_wait(&h->cond[PARSE2], &h->lock[PARSE2]);
			h->slice_cnt--;
			s= &h->slices[h->slice_next++];
			h->slice_next %= MAX_SLICE_COUNT;
			pthread_mutex_unlock(&h->lock[PARSE2]);
		}*/
		ff_init_slice(n, s);
		reset_h264mb(s, n->mb_width, n->mb_height);
		for(int i=0; i<MAX_PIC_COUNT; i++){
			if(h->picture[i].reference==0){
				pic= &h->picture[i];
				break;
			}
		}
// 		{
// 			pthread_mutex_lock(&h->lock[PARSE3]);
// 			while (h->free_pic_cnt<=0)
// 				pthread_cond_wait(&h->cond[PARSE3], &h->lock[PARSE3]);
// 			h->free_pic_cnt--;
// 			/* use first free picture */
// 			for(int i=0; i<MAX_PIC_COUNT; i++){
// 				if(h->picture[i].reference==0){
// 					pic= &h->picture[i];
// 					break;
// 				}
// 			}
// 			pthread_mutex_unlock(&h->lock[PARSE3]);
// 		}
		ff_alloc_picture(n, s, pic);

		decode_nal_units(n, s, frm, pic);


		decode_slice_entropy(&hcabac, &cabac, s);
		memcpy( s2, s, sizeof(MBSlice)); //this only copys the COMMON_SLICE part
		av_freep(&s->gb.raw);
		decode_slice_mb_seq(d, s2);

//         if (s2->release_cnt>0) {
//             int i;
//             for (i=0; i<s2->release_cnt; i++){
//                 if ((s2->release_ref[i]->reference & ~2) == 0)
//                     default_release_buffer(h, s2->release_ref[i]);
//                 else
//                     s2->release_ref[i]->reference &= ~2;
//             }
//             s->release_cnt=0;
//         }

if (s->release_cnt>0) {
	int i;
	for (i=0; i<s->release_cnt; i++){
		s->release_ref[i]->reference &= ~2;
	}
	s->release_cnt=0;
}


        {
			pthread_mutex_lock(&h->lock[PARSE2]);
			h->slice_cnt++;
			pthread_cond_signal(&h->cond[PARSE2]);
			pthread_mutex_unlock(&h->lock[PARSE2]);
		}

		out =output_frame(w, s2->current_picture, h->ofile, h->width, h->height);
		print_report(w->frame_number, w->video_size, 0);

		if (out){
// 			if ((out->reference & ~1) == 0)
// 				default_release_buffer(h, out);
// 			else
				out->reference &= ~1;
		}

		{
			pthread_mutex_lock(&h->lock[ENTROPY]);
			h->ed_cnt--;
			pthread_cond_signal(&h->cond[ENTROPY]);
			pthread_mutex_unlock(&h->lock[ENTROPY]);
		}
	}
	while (output_frame(w, NULL, h->ofile, h->width, h->height));
	print_report(w->frame_number, w->video_size, 1);

	av_free(w->bit_buffer);

	{//propagate exit
		pthread_mutex_lock(&h->lock[WRITE]);
		while (h->write_cnt>= MAX_DELAYED_PIC_COUNT)
			pthread_cond_wait(&h->cond[WRITE], &h->lock[WRITE]);
		last_pic.reference = -1;
		h->write_q[h->write_fi] = &last_pic;
		h->write_cnt++;
		h->write_fi++; h->write_fi %= MAX_DELAYED_PIC_COUNT;
		pthread_cond_signal(&h->cond[WRITE]);
		pthread_mutex_unlock(&h->lock[WRITE]);

	}
	free_cabac(&hcabac);

	pthread_exit(NULL);
	return NULL;

}
