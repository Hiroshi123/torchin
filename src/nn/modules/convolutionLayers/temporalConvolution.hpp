
#include "module.hpp"

namespace nn {

  template<typename T>
  class TemporalConvolution : public nn::Module {
    
    size_t nInputPlane;
    size_t nOutputPlane;
    size_t kW,dW,padW;
    
    Tensor<T> weight;
    Tensor<T> delta_weight;
    Tensor<T> bias;    
    Tensor<T> delta_bias;
    Tensor<T> buffer;
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    void accGradParameters();
    
    void reset();
    void name() const;
    
  protected:
    
  public:
    
    TemporalConvolution
    (const size_t nInputPlane, const size_t nOutputPlane, const size_t kw = 3,
     const size_t dw = 1, const size_t padw = 0);
    
    // (const size_t nInputPlane, const size_t nOutputPlane, size_t kW = 3,
    //  size_t kH = 3, size_t dW = 1, size_t dH = 1, size_t padW = 0, size_t padH = 0);
    
  };
  
  template<typename T>
  TemporalConvolution<T>::TemporalConvolution
  (const size_t ninputplane, const size_t noutputplane, const size_t kw ,
   const size_t dw, const size_t padw)
  {
    
    nInputPlane = ninputplane;
    nOutputPlane = noutputplane;
    kW = kw;dW = dw;
    
    //weight.resize({nOutputPlane,kH,kW,nInputPlane});
    //bias.resize({kW,kH,nInputPlane});
    
    reset();
      
  };

  template<typename T>
  void TemporalConvolution<T>::reset(){
    
    //bias->print();
    //double randomSeed = 100;
    //std::mt19937 randomEngine(randomSeed);
    //std::uniform_real_distribution<double> randomDoubleDistribution(0.0, 1.0);

    weight.fill(1);
    //weight.rand();
    bias.rand();
    
  };

  
  template<typename T>
  Tensor<T>& TemporalConvolution<T>::updateOutput(Tensor<T>& input){
    
    buffer.resize({
	input.shape.dimensions[0],
	  input.shape.dimensions[1],	  
	  nOutputPlane});
    
    input.convolution3D(weight,buffer);
    return buffer;
  };
  
};

