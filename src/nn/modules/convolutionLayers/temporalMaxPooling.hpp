

#include "module.hpp"

namespace nn {

  template<typename T>
  class TemporalMaxPooling : public nn::Module {
    
    size_t nFeature;
    
    size_t kW,dW,padW;
    
    Tensor<T> buffer;
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    //void accGradParameters();
    //void reset();
    void name() const;
    
  protected:
    
  public:
    
    TemporalMaxPooling
    (const size_t nfeature, const size_t kw = 2 ,
     const size_t dw = 2, const size_t padw = 0 );
    
  };
  
  template<typename T>
  TemporalMaxPooling<T>::TemporalMaxPooling
  (const size_t nfeature, const size_t kw,
   const size_t dw,const size_t padw)
  {
    
    nFeature = nfeature;
    kW = kw;
    dW = dw;
    padW = padw;
    
  };
  
  template<typename T>
  void TemporalMaxPooling<T>::name() const {
    std::cout << "TemporalMaxPooling\n";
  };
  
  
  template<typename T>
  Tensor<T>& TemporalMaxPooling<T>::updateOutput(Tensor<T>& input){
    
    buffer.resize({
	input.shape.dimensions[0] / kW + padW,
	  nFeature });
    
    //input.maxPooling2DwithF(buffer,kW,kH,dW,dH,padW,padH);
    
    return buffer;
  };

  template<typename T>
  Tensor<T>& TemporalMaxPooling<T>::updateGradInput(Tensor<T>& input){
    
    buffer.resize({
	input.shape.dimensions[0] / kW + padW,
	  nFeature });
    
    //input.convolution3D(weight,buffer);
    return buffer;
  };
  
};








