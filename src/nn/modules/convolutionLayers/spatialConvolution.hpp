
#include "module.hpp"

namespace nn {
  
  template<typename T>
  class SpatialConvolution : public nn::Module {
    
    size_t nInputPlane;
    size_t nOutputPlane;
    size_t kW,kH,dW,dH,padW,padH;
    
    Tensor<T> weight;
    Tensor<T> delta_weight;
    Tensor<T> bias;    
    Tensor<T> delta_bias;
    Tensor<T> buffer;
    
    Tensor<T>& updateOutput(Tensor<T>&) override;
    Tensor<T>& updateGradInput(Tensor<T>& i) override;
    void accGradParameters() override {};
    
    void reset();
    void name() const;
    
  protected:
    
    
    
  public:
    
    SpatialConvolution
    (const size_t nInputPlane, const size_t nOutputPlane, size_t kW = 3,
     size_t kH = 3, size_t dW = 1, size_t dH = 1, size_t padW = 0, size_t padH = 0);
    
    //Tensor<T> weight;
    
  };
  
  template<typename T>
  SpatialConvolution<T>::SpatialConvolution
  (const size_t ninputplane, const size_t noutputplane, const size_t kw,
   const size_t kh, const size_t dw, const size_t dh, const size_t padw, const size_t padh)
  {
    
    nInputPlane = ninputplane;
    nOutputPlane = noutputplane;
    
    kW = kw;kH = kh;dW = dw;dH = dh;
    
    //weight.resize({kH,kW,nOutputPlane,nInputPlane});
    weight.resize({nOutputPlane,kH,kW,nInputPlane});
    
    bias.resize({kW,kH,nInputPlane});
    
    reset();
    
  };
  
  template<typename T>
  Tensor<T>& SpatialConvolution<T>::updateOutput(Tensor<T>& input){
    
    buffer.resize({
	input.shape.dimensions[0],input.shape.dimensions[1],	  
	  nOutputPlane});
    
    input.convolution3D(weight,buffer);
    
    //buffer.print();
    
    return buffer;
  };

  template<typename T>
  Tensor<T>& SpatialConvolution<T>::updateGradInput(Tensor<T>& input) {
    
    buffer.resize(input.shape.dimensions);
    input.convolution3D(weight,buffer);
    return buffer;
    
  };
  
  
  template<typename T>
  void SpatialConvolution<T>::reset(){
    
    //bias->print();
    //double randomSeed = 100;
    //std::mt19937 randomEngine(randomSeed);
    //std::uniform_real_distribution<double> randomDoubleDistribution(0.0, 1.0);

    weight.fill(1);
    //weight.rand();
    bias.rand();
    
  };

  template<typename T>
  void SpatialConvolution<T>::name() const{
    
    std::cout << "SpatialConvolution" << std::endl;
  };
  
};
