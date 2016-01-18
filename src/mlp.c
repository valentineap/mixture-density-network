#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#define NEURON_TYPE_LINEAR ((const unsigned char) 'A')
#define NEURON_TYPE_TANH ((const unsigned char) 'B')

typedef struct mlp_settings {
  int nin;
  int nout;
  int number_of_hidden_layers;
  int * neurons_per_hidden_layer;
  unsigned char * neuron_types;
} mlp_settings_t;

typedef struct layer {
  int nin;
  int nout;
  double * w;
  double * b;
  unsigned char neuron_type;
} layer_t;

typedef struct mlp {
  int nlayers;
  layer_t layers[];
} mlp_t;

/////////////////////////////
// Random number functions //
/////////////////////////////
volatile int seeded_random=0;//Has random seed been initialised?
#ifdef ACML
int lstate=633;
int state[633];
void random_uniform_mem(double *start,int n, double lo,double hi){
  //Fill memory block with uniformly-distributed random numbers
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  dranduniform(n,lo,hi,state,start,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
}

void random_normal_mem(double *start, int n, double stdev){
  //Fill memory block with gaussian-distributed random numbers
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  drandgaussian(n,0.0,stdev,state,start,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
}

inline double random_uniform(double lo,double hi){
  //Return a random number from a uniform distribution in [lo,hi]
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  double r;
  dranduniform(1,lo,hi,state,&r,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
  return r;
}

inline double random_normal(){
  //Return a random number from a Gaussian with unit stdev
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  double r;
  drandgaussian(1,0.,1.0,state,&r,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
  return r;
}

#else
int have_stored_rand=0;
double stored_rand;

inline double random_normal(){
  //Return a random number drawn from a Gaussian distribution with standard deviation 1.0
  //Marsaglia's Polar Method
  double u,v,x1,x2,w;
#ifdef NATIVE_RAND
  int seedval;
  if(seeded_random==0){
    seedval=(unsigned)time(NULL)+getpid();
    srand(seedval);
    seeded_random=1;
  }
#else //Mersenne Twister
  unsigned long seedval;
  if(seeded_random==0){
    seedval=(unsigned long)time(NULL)+getpid();
    init_genrand(seedval);
    seeded_random=1;
  }
#endif
  if(have_stored_rand){
    have_stored_rand=0;
    return stored_rand;
  } else {
    do{
#ifdef NATIVE_RAND
      u=(double)rand()/(double)RAND_MAX;
      v=(double)rand()/(double)RAND_MAX;
#else
      u=genrand_real1();
      v=genrand_real2();
#endif
      x1=2.0*u-1.0;
      x2=2.0*v-1.0;
      w=x1*x1+x2*x2;
    } while (w>=1.0);
    stored_rand=x2*sqrtl((-2.0*logl(w))/w);
    have_stored_rand=1;
    return x1*sqrtl((-2.0*logl(w))/w);
  }
}
#endif

layer_t layer_allocate(int nin, int nout, unsigned char neuron_type){
  layer_t l;
  l.nin = nin;
  l.nout = nout;
  if((l.w = malloc(nin*nout*sizeof(double))) == NULL){
    printf("Error allocating memory\n");
    abort();
  }
  if((l.b = malloc(nout*sizeof(double)))==NULL){
    printf("Error allocating memory\n");
    abort();
  }
  l.neuron_type = neuron_type;
  return l;
}

void layer_free(layer_t layer){
  free(layer.w);
  free(layer.b);
}

mlp_t * mlp_allocate(mlp_settings_t settings){
  mlp_t * net = malloc(sizeof(mlp_t)+(1+settings.number_of_hidden_layers)*sizeof(layer_t));
  net->nlayers = 1+settings.number_of_hidden_layers;
  int i,nin,nout;
  nin = settings.nin;
  nout = *(settings.neurons_per_hidden_layer);
  for(i=0;i<settings.number_of_hidden_layers;i++){
    nout = *(settings.neurons_per_hidden_layer+i);
    net->layers[i] = layer_allocate(nin,nout,*((settings.neuron_types)+i));
    nin = nout;
  }
  net->layers[settings.number_of_hidden_layers+1] = layer_allocate(nin,settings.nout,*(settings.neuron_types+settings.number_of_hidden_layers+1));
  return net;
}

void mlp_free(mlp_t * net){
  int i;
  for(i=0;i<net->nlayers;i++){
    layer_free(net->layers[i]);
  }
}

mlp_evaluate(mlp_t * net, int negs, double * inputs,double * outputs){
  int ilayer;
  double * lin = malloc(negs*net->layers[0].nin*sizeof(double));
  memcpy(inputs,lin,negs*net->layers[0].nin*sizeof(double));
  for(ilayer=0; ilayer<net->nlayers; ilayer++){
    double * lout = malloc(negs*net->layers[ilayer].nout*sizeof(double));
    for()

}

int main(int argc, char **argv){
  mlp_settings_t settings;
  settings.nin = 10;
  settings.nout = 3;
  settings.number_of_hidden_layers = 1;
  settings.neurons_per_hidden_layer = malloc(sizeof(int));
  settings.neuron_types = malloc(2*sizeof(unsigned char));
  *(settings.neurons_per_hidden_layer) = 10;
  *(settings.neuron_types) = NEURON_TYPE_TANH;
  *(settings.neuron_types+1) = NEURON_TYPE_LINEAR;

  mlp_t * net = mlp_allocate(settings);
  mlp_free(net);
}
