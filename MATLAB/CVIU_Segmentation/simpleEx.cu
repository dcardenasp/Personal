__global__ void kGauss( double * Y, const double * x, const double * w, const int K )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double y = Y[idx];
    Y[idx]=0;   
    for(int i=0; i<K; ++i)
      Y[idx] += w[i]*exp(-(y-x[i])*(y-x[i]));  
}
