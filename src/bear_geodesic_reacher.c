/*
 * bear_geodesic_reacher.c -- BearRL env #1: quaternion geodesic reacher
 *
 * THE MATH (from the 2026-08-24 Kevin-Bacon research, docs/research/):
 *   A P-frame in the VHF engine = transport of a frame latent q_t to q_{t+1}
 *   along a learned vector field on SO(3) (unit quaternions). Before we learn
 *   the field, the PROOF ENVIRONMENT must verify the substrate:
 *
 *     - geodesic distance d(q0,q1) = ||log(q0^-1 * q1)|| on S^3
 *     - exp/log round trip: log(exp(v)) == v for |v| < pi
 *     - slerp(t) traces the geodesic: d(q0, slerp(t)) = t*d(q0,q1)
 *     - parallel-transported tangent vectors stay tangent
 *
 *   The RL formulation: state = current unit quaternion; action = tangent
 *   (angular velocity) step; reward = -geodesic distance to target. An agent
 *   following the NEGATIVE GRADIENT of geodesic distance IS integrating the
 *   shortest-path ODE -- this is the null-model any learned flow field must
 *   beat. We train it here with a tiny policy-gradient loop (REINFORCE with
 *   baseline), pure C11, no deps.
 *
 * Emits JSON metrics to stdout every REPORT_EVERY episodes -> the dashboard
 * reads these. Certificate = final solved rate + committed logs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
static const float gamma_c=0.95f;
float g_lr=0.02f;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- minimal quaternion ops (mirrors wubu_quaternion_ops.c) ----- */
static void q_normalize(float *q){
    float n = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(n<1e-12f){q[0]=1;q[1]=q[2]=q[3]=0;return;}
    q[0]/=n;q[1]/=n;q[2]/=n;q[3]/=n;
}
static void q_mul(float *out,const float*a,const float*b){ /* Hamilton */
    out[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    out[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    out[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    out[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}
static void q_conj(float*o,const float*q){o[0]=q[0];o[1]=-q[1];o[2]=-q[2];o[3]=-q[3];}

/* log of unit quat -> pure quat (shortest arc); returns angle in [0,pi] */
static float q_log(float*v3,const float*q){
    float vnorm=sqrtf(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    float ang=2.0f*atan2f(vnorm,q[0]); /* in [0, 2pi) via atan2 */
    if(ang>(float)M_PI) ang-=2.0f*(float)M_PI;      /* wrap: geodesic <= pi */
    if(vnorm>1e-9f){
        float s=ang/vnorm; /* signed half-angle scaled */
        v3[0]=q[1]*s/2.0f*2.0f; /* = q_i * (ang/vnorm) ... keep as axis*angle/...*/
        v3[0]=q[1]*(ang/(vnorm>0?vnorm:1));
        v3[1]=q[2]*(ang/vnorm);
        v3[2]=q[3]*(ang/vnorm);
        /* NOTE: v3 is axis*angle (full rotation vector, |v3|=|ang|) */
    } else { v3[0]=v3[1]=v3[2]=0; }
    return fabsf(ang);
}
/* exp of pure rotation vector v3 (|v3|<=pi) -> unit quat */
static void q_exp(float*q,const float*v3){
    float th=sqrtf(v3[0]*v3[0]+v3[1]*v3[1]+v3[2]*v3[2]);
    if(th>1e-9f){
        float s=sinf(th/2.0f)/th;
        q[0]=cosf(th/2.0f);
        q[1]=v3[0]*s;q[2]=v3[1]*s;q[3]=v3[2]*s;
    }else{
        q[0]=1;q[1]=v3[0]/2;q[2]=v3[1]/2;q[3]=v3[2]/2;
    }
    q_normalize(q);
}
static float q_geodesic(const float*a,const float*b){
    float abinv[4],d[4],v3[3];
    q_conj(abinv,a); q_mul(d,abinv,b);
    return q_log(v3,d);
}

/* ---------------- policy: linear in features, softmax over K actions --- */
#define K 8                 /* angular velocity primitives + null action   */
static float ACT[K][3]={
    { .25,0,0},{-.25,0,0},{0, .25,0},{0,-.25,0},{0,0, .25},{0,0,-.25},
    { .12,.12,.12},{-.12,-.12,-.12}
};
/* adaptive null action removed: agent needs pure rotation primitives; fine
   steps emerge from repeated small rotations + early termination */
typedef struct{ float W[K][6]; float b[K]; } Policy;

int g_greedy=0;
static int g_best_action(const Policy*P,const float*f){
    float best=-1e30f; int bi=0;
    for(int k=0;k<K;k++){
        float s=P->b[k];
        for(int i=0;i<6;i++) s+=P->W[k][i]*f[i];
        if(s>best){best=s;bi=k;}
    }
    return bi;
}
   /* feats: err vec + cross */

static void features(float *f,const float *cur,const float *tgt){
    /* tangent-space error direction at cur (local frame) */
    float ci[4],d[4],v3[3];
    q_conj(ci,cur); q_mul(d,ci,tgt);
    q_log(v3,d);
    f[0]=v3[0];f[1]=v3[1];f[2]=v3[2];
    f[3]=cur[0];f[4]=cur[1];f[5]=cur[2];
}
static void softmax(float*p,const Policy*P,const float*f,float temp){
    float m=-1e30f,z=0;
    for(int k=0;k<K;k++){
        float s=P->b[k];
        for(int i=0;i<6;i++) s+=P->W[k][i]*f[i];
        p[k]=s/temp; if(p[k]>m)m=p[k];
    }
    for(int k=0;k<K;k++){ p[k]=expf(p[k]-m); z+=p[k]; }
    for(int k=0;k<K;k++) p[k]/=z;
}

/* ---------------- RNG --------------------------------------------------*/
static unsigned long rs=88172645463325252UL;
static float urand(void){ /* xorshift -> [0,1) */
    rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;
    return (float)((rs>>11)&0x3FFFFFFF)/(float)0x3FFFFFFF;
}

/* ---------------- one episode ------------------------------------------*/
typedef struct{
    float reward; int steps; int solved; float final_dist;
} EpResult;

static void run_episode(Policy*P,float temp,int max_steps,
                        const float*target,float tol,EpResult*out)
{
    float cur[4]={1,0,0,0};
    float R=0; int solved=0, T=0;
    float last_prev=q_geodesic(cur,target);
    (void)last_prev;

    static float probs_hist[64][K];
    static int   acts_hist[64];
    static float rews[64], rets[64];
    if(max_steps>64) max_steps=64;

    for(int t=0;t<max_steps;t++){
        float f[6],p[K];
        features(f,cur,target);
        softmax(p,P,f,temp);
        int a=K-1; float r=urand(),acc=0;
        for(int k=0;k<K;k++){acc+=p[k];if(r<=acc){a=k;break;}}
        memcpy(probs_hist[t],p,sizeof(float)*K); acts_hist[t]=a;
        if(g_greedy) a=acts_hist[t]=g_best_action(P,f);

        float dq[4]; q_exp(dq,ACT[a]);
        float nxt[4]; q_mul(nxt,cur,dq); q_normalize(nxt);

        float prev_dist = (t==0)? q_geodesic(cur,target) : last_prev;
        float dist=q_geodesic(nxt,target); last_prev=dist;
        float r_t = 3.0f*(prev_dist-dist);      /* potential-based shaping */
        if(dist<tol && !solved){ r_t += 10.0f; solved=1; }
        prev_dist=dist;
        rews[t]=r_t; R+=r_t; T=t+1;

        cur[0]=nxt[0];cur[1]=nxt[1];cur[2]=nxt[2];cur[3]=nxt[3];
        if(dist<tol) break;
    }

    /* discounted returns */
    float ret=0;
    for(int t=T-1;t>=0;t--){ ret=rews[t]+gamma_c*ret; rets[t]=ret; }
    float base=rets[0]/(float)T;

    /* policy-gradient replay */
    float gradW[K][6]={{0}},gradb[K]={0};
    float c2[4]={1,0,0,0};
    for(int t=0;t<T;t++){
        int a=acts_hist[t];
        float adv=rets[t]-base;
        float f[6],p[K];
        features(f,c2,target);
        softmax(p,P,f,temp);
        for(int k=0;k<K;k++){
            float g=((k==a?1-p[k]:-p[k]))*adv;
            gradb[k]+=g;
            for(int i=0;i<6;i++) gradW[k][i]+=g*f[i];
        }
        float dq[4]; q_exp(dq,ACT[a]);
        float nx[4]; q_mul(nx,c2,dq); q_normalize(nx);
        c2[0]=nx[0];c2[1]=nx[1];c2[2]=nx[2];c2[3]=nx[3];
    }
    for(int k=0;k<K;k++){
        P->b[k]+=g_lr*gradb[k];
        for(int i=0;i<6;i++) P->W[k][i]+=g_lr*gradW[k][i];
    }
    out->reward=R; out->steps=T; out->solved=solved;
    out->final_dist=q_geodesic(cur,target);
}

int main(int argc,char**argv){
    long episodes = argc>1?atol(argv[1]):60000;
    int max_steps = argc>2?atoi(argv[2]):24;
    float tol = argc>3?(float)atof(argv[3]):0.05f;
    const int REPORT_EVERY=100;

    Policy P; memset(&P,0,sizeof(P));
    srand((unsigned)time(NULL));

    fprintf(stderr,"# bear geodesic reacher | episodes=%ld max_steps=%d tol=%g\n",
            episodes,max_steps,tol);

    float ema_reward=0,ema_dist=0; long solved_total=0,solved_window=0;
    clock_t t0=clock();

    for(long ep=1;ep<=episodes;ep++){
        /* random target on S^3 */
        float tgt[4];
        do{ tgt[0]=urand()*2-1;tgt[1]=urand()*2-1;
            tgt[2]=urand()*2-1;tgt[3]=urand()*2-1; }while(
            tgt[0]*tgt[0]+tgt[1]*tgt[1]+tgt[2]*tgt[2]+tgt[3]*tgt[3]>1||
            tgt[0]*tgt[0]+tgt[1]*tgt[1]+tgt[2]*tgt[2]+tgt[3]*tgt[3]<1e-4);
        q_normalize(tgt);

        /* anneal temperature */
        float temp = 2.0f/(1.0f+(float)ep/2000.0f)+0.15f;

        EpResult r; run_episode(&P,temp,max_steps,tgt,tol,&r);
        ema_reward = 0.98f*ema_reward + 0.02f*r.reward;
        ema_dist   = 0.98f*ema_dist   + 0.02f*r.final_dist;
        solved_total+=r.solved;
        if(ep>episodes-REPORT_EVERY) solved_window+=r.solved;

        if(ep%REPORT_EVERY==0){
            double secs=(double)(clock()-t0)/CLOCKS_PER_SEC;
            printf("{\"type\":\"metrics\",\"episode\":%ld,"
                   "\"ema_reward\":%.4f,\"ema_final_dist\":%.5f,"
                   "\"solve_rate\":%.4f,\"temp\":%.3f,\"wall_s\":%.2f}\n",
                   ep,ema_reward,ema_dist,(float)solved_total/ep,temp,secs);
            fflush(stdout);
        }
    }

    /* greedy eval phase */
    g_greedy=1;
    { int gsolved=0, N=2000; float gdist=0;
      for(int i=0;i<N;i++){
        float tgt[4];
        do{ tgt[0]=urand()*2-1;tgt[1]=urand()*2-1;tgt[2]=urand()*2-1;tgt[3]=urand()*2-1; }
        while(tgt[0]*tgt[0]+tgt[1]*tgt[1]+tgt[2]*tgt[2]+tgt[3]*tgt[3]>1||
              tgt[0]*tgt[0]+tgt[1]*tgt[1]+tgt[2]*tgt[2]+tgt[3]*tgt[3]<1e-4);
        q_normalize(tgt);
        EpResult r2; run_episode(&P,0.001f,max_steps,tgt,tol,&r2);
        gsolved+=r2.solved; gdist+=r2.final_dist;
      }
      printf("{\"type\":\"greedy_eval\",\"episodes\":%d,\"solve_rate\":%.4f,\"mean_final_dist\":%.5f}\n",
             N,(float)gsolved/N,gdist/N);
    }
    printf("{\"type\":\"certificate\",\"episodes\":%ld,\"final_solve_rate\":%.4f,"
           "\"final_ema_dist\":%.6f,\"tol\":%.4f,\"status\":\"%s\"}\n",
           episodes,(float)solved_window/REPORT_EVERY,ema_dist,tol,
           ((float)solved_window/REPORT_EVERY>0.90f)?"SOLVED":"TRAINING");
    return 0;
}
