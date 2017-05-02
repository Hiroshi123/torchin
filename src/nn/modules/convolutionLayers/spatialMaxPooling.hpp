

#include "module.hpp"

namespace nn {

  template<typename T>
  class SpatialMaxPooling : public nn::Module {
    
    size_t nFeature;
    
    size_t kW,kH,dW,dH,padW,padH;
    
    Tensor<T> buffer;
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    //void accGradParameters();
    //void reset();
    void name() const;
    
  protected:
    
  public:
    
    SpatialMaxPooling
    (const size_t nfeature, const size_t kw = 2 , const size_t kh = 2,
     const size_t dw = 2,const size_t dh = 2, const size_t padw = 0, const size_t padh = 0);
    
  };
  
  template<typename T>
  SpatialMaxPooling<T>::SpatialMaxPooling
  (const size_t nfeature, const size_t kw, const size_t kh,
   const size_t dw,const size_t dh, const size_t padw, const size_t padh)
  {
    
    nFeature = nfeature;
    kW = kw;kH = kh;
    dW = dw;dH = dh;
    padW = padw; padH = padh;
    
  };
  
  template<typename T>
  void SpatialMaxPooling<T>::name() const {
    std::cout << "SpatialMaxPooling\n";
  };
  
  
  template<typename T>
  Tensor<T>& SpatialMaxPooling<T>::updateOutput(Tensor<T>& input){
    
    buffer.resize({
	input.shape.dimensions[0] / kW + padW,
	  input.shape.dimensions[1] / kH + padH,	  
	  nFeature });
    
    input.maxPooling2DwithF(buffer,kW,kH,dW,dH,padW,padH);
    
    return buffer;
  };

  template<typename T>
  Tensor<T>& SpatialMaxPooling<T>::updateGradInput(Tensor<T>& input){
    
    buffer.resize({
	input.shape.dimensions[0] / kW + padW,
	  input.shape.dimensions[1] / kH + padH,	  
	  nFeature });
    
    //input.convolution3D(weight,buffer);
    return buffer;
  };
  
};




