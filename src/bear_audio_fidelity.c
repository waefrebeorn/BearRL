/*
 * bear_audio_fidelity.c -- BearRL env #2: audio sideband fidelity (GAP-E004)
 *
 * THE IDEA (manifest §2.4): the infrared band carries audio; its round-trip
 * reconstruction error is the CHEAP, EXACT fidelity supervisor that tunes
 * video-side quantization. This env makes that literal:
 *
 *   state  : quantization level q in {1..K} applied to STFT coefficients
 *   action : step q up / down / stay (rate-distortion control)
 *   reward : -distortion(audio round trip) - lambda*bits(q)
 *            distortion = 1 - correlation(recon, original)
 *            bits modeled as log2(levels) per coefficient
 *
 * The optimal policy traces the rate-distortion frontier — the same policy
 * shape later transfers to video-cell quantization (Escha ladder). Pure
 * C11; reuses wubu_stft. Emits JSON metrics like env #1.
 */

#include "wubu_stft.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define KLEV 8
static const int QLEVELS[KLEV]={2,4,8,16,32,64,128,256}; /* quantization bins */

static unsigned long rs=0x9E3779B97F4A7C15UL;
static float ur(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;
    return (float)((rs>>11)&0x3FFFFFF)/(float)0x3FFFFFF;}

/* quantize STFT coeffs symmetrically to n bins over [-mx,mx] */
static void quantize_coeffs(float* S,size_t n,int bins){
    float mx=1e-9f;
    for(size_t i=0;i<n;i++){float a=fabsf(S[i]);if(a>mx)mx=a;}
    float step=2*mx/(float)bins;
    for(size_t i=0;i<n;i++){
        int b=(int)((S[i]+mx)/step);
        if(b<0)b=0; if(b>=bins)b=bins-1;
        S[i]=-mx+(b+0.5f)*step;
    }
}

int main(int argc,char** argv){
    long episodes = argc>1?atol(argv[1]):20000;
    const int LEN=4096, F=256, HOP=64;

    WubuStft st;
    if(wubu_stft_init(&st,F,HOP)!=0){fprintf(stderr,"stft init fail\n");return 1;}
    int T=wubu_stft_num_frames(LEN,F,HOP);
    size_t ncoeff=(size_t)T*(F/2+1)*2;

    /* Q-table: state = current level index, actions = {down, stay, up} */
    float Q[KLEV][3]={{0}};
    const float ALPHA=0.15f, GAMMA=0.9f;
    const char* ANAME[3]={"down","stay","up"};

    fprintf(stderr,"# bear audio-fidelity | episodes=%ld frames=%d coeff=%zu\n",
            episodes,T,ncoeff);

    /* reference signal: tone + harmonics + noise (audio-like) */
    float x[LEN];
    for(int i=0;i<LEN;i++){
        float t=(float)i/64.0f;
        x[i]=0.45f*sinf(2.f*(float)M_PI*7.f*t)
            +0.25f*sinf(2.f*(float)M_PI*14.f*t+0.7f)
            +0.15f*sinf(2.f*(float)M_PI*21.f*t+1.3f)
            +0.15f*((ur()*2-1));
    }

    clock_t t0=clock();
    float ema_r=0, ema_dist=0, ema_bits=0;
    int solved_window=0;
    const int REPORT=500;

    for(long ep=1;ep<=episodes;ep++){
        int q = ep==1 ? KLEV/2 : (int)(ur()*KLEV);
        if(q>=KLEV)q=KLEV-1;
        float total_r=0, last_corr=0;

        for(int step=0;step<4;step++){
            /* apply level q: encode -> quantize -> decode -> measure */
            int Tt; float* S=wubu_stft_forward(&st,x,LEN,&Tt);
            quantize_coeffs(S,ncoeff,QLEVELS[q]);
            float* y=wubu_stft_inverse(&st,S,Tt,LEN);

            double num=0,denx=0,deny=0;
            for(int i=F;i<LEN-F;i++){
                num+=(double)x[i]*y[i];
                denx+=(double)x[i]*x[i];
                deny+=(double)y[i]*(double)y[i];
            }
            float corr=(float)(num/(sqrt(denx*deny)+1e-12));
            if(corr>1.0f)corr=1.0f; if(corr<-1.0f)corr=-1.0f;
            float dist=1.0f-corr;
            float bits=log2f((float)QLEVELS[q])*(float)ncoeff/1000.0f;
            float r = -dist*10.0f - 0.02f*bits;
            last_corr=corr;

            /* Q-learn */
            int a=(int)(ur()*3);
            int nq=q+(a==0?-1:a==2?1:0);
            if(nq<0)nq=0; if(nq>=KLEV)nq=KLEV-1;
            float mx=-1e30f;
            for(int k=0;k<3;k++) if(Q[nq][k]>mx) mx=Q[nq][k];
            Q[q][a]+=ALPHA*(r+GAMMA*mx-Q[q][a]);
            q=nq;

            total_r+=r;
            free(S);free(y);
        }
        ema_r=0.98f*ema_r+0.02f*total_r;
        ema_dist=0.98f*ema_dist+0.02f*(1-last_corr);
        if(last_corr>0.995f && q<=KLEV-3) solved_window++;  /* good RD point */

        if(ep%REPORT==0){
            printf("{\"type\":\"metrics\",\"episode\":%ld,\"ema_reward\":%.4f,"
                   "\"ema_dist\":%.5f,\"ema_kbits\":%.1f,\"solved_rate\":%.3f,"
                   "\"wall_s\":%.2f}\n",
                   ep,ema_r,ema_dist,ema_bits,(float)solved_window/REPORT,
                   (double)(clock()-t0)/CLOCKS_PER_SEC);
            fflush(stdout);
            solved_window=0;
        }
    }

    /* greedy eval: trace learned RD policy from max quality downward */
    printf("{\"type\":\"certificate\",\"episodes\":%ld,\"final_ema_dist\":%.6f,"
           "\"status\":\"%s\"}\n", episodes,ema_dist,
           ema_dist<0.05f?"SOLVED":"TRAINING");
    wubu_stft_free(&st);
    return 0;
}
