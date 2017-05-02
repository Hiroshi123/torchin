
#include "module.hpp"

namespace nn {
  
  template<typename T>
  class SpatialConvolutions : public nn::Module {
    
    size_t nInputPlane;
    size_t nOutputPlane;
    size_t kW,kH,dW,dH,padW,padH;
    
    Tensor<T> weight;
    Tensor<T> delta_weight;
    Tensor<T> bias;    
    Tensor<T> delta_bias;
    
    Tensor<T> buffer;
    
  protected:
    
    Tensor<T>& updateOutput(Tensor<T>&);
    //Tensor<T>& updateGradInput(Tensor<T>&);
    //void accGradParameters();
    
  public:
    
    SpatialConvolutions
    (const size_t nInputPlane, const size_t nOutputPlane, size_t kW = 3,
     size_t kH = 3, size_t dW = 1, size_t dH = 1, size_t padW = 0, size_t padH = 0);
    
    //Tensor<T> weight;
    
    void reset();
    void name() const;
    
  };
  
  template<typename T>
  SpatialConvolutions<T>::SpatialConvolutions
  (const size_t ninputplane, const size_t noutputplane, const size_t kw,
   const size_t kh, const size_t dw, const size_t dh, const size_t padw, const size_t padh)
  {
    
    nInputPlane = ninputplane;
    nOutputPlane = noutputplane;
    
    kW = kw;kH = kh;dW = dw;dH = dh;
    
    weight.resize({kH,kW,nOutputPlane,nInputPlane});
    
    //weight.resize({3,4});
    //bias.resize({kW,kH,nInputPlane});
    
    // reset();
    
  };

  template<typename T>
  Tensor<T>& SpatialConvolutions<T>::updateOutput(Tensor<T>& input){
    
    //assert(typeid(float) == typeid(T));
    
    //std::cout  << "come**" << std::endl;
    
    buffer.resize(input.shape.dimensions);
    
    //for(size_t o = 0 ; o < nOutputPlane ; o++ )
    T stock = 0;
    
    T resW = 10;
    T resH = 10;
    
    // for(size_t o = 0 ; o < nOutputPlane ; o ++ ){
    //   for(size_t iw = 0 ; iw < resW ; iw ++ ){
    // 	for(size_t ih = 0 ; ih < resH ; ih ++ ){
    // 	  for (size_t w = 0 ; w < kW ; w ++ ){
    // 	    for (size_t h = 0 ; h < kH ; h ++ ){
    // 	      for (size_t d = 0 ; d < nInputPlane ; d++ ){
    // 		// weight->data[0] = 1;
    // 		stock += 
    // 		  input.data[ d + h * nInputPlane + w * nInputPlane * kH + ih * nInputPlane + iw * nInputPlane * resH + o * nInputPlane * resW * resH]
    // 		  * weight.data[ d + h * nInputPlane + w * nInputPlane * kH ];
    // 	      }
    // 	    }
    // 	  }
    // 	  buffer[ iw + ih * resW ] += stock;
    // 	}
    //   }
    // }
    
    return buffer;
  };
  
  template<typename T>
  void SpatialConvolutions<T>::reset(){
    
    //bias->print();
    //double randomSeed = 100;
    //std::mt19937 randomEngine(randomSeed);
    //std::uniform_real_distribution<double> randomDoubleDistribution(0.0, 1.0);
    
    weight.rand();
    bias.rand();
    
  };

  template<typename T>
  void SpatialConvolutions<T>::name() const{
    
    std::cout << "SpatialConvolution" << std::endl;
  };
  
};
