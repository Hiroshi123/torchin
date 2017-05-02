
#include "module.hpp"

namespace nn{
  
  template<typename T>
  class HardShrink : public nn::Module {

    Tensor<T> output;
    
  public:
    
    HardShrink() = default;
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    
    std::function<T(T)> operation = [](T x) -> T {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));};
    std::function<T(T,T)> grad = [](T x,T y) -> T {return x * ( 1 - y * y );};
    
  };
  
  //template<typename T> HardShrink<T>::HardShrink(){};
  
  template<typename T>
  Tensor<T>& HardShrink<T>::updateOutput(Tensor<T>& input){
    
    input.apply(operation);
    
    output.resize(input.shape.dimensions);
    input.copy(output);
    return input;
    
  };

  template<typename T>
  Tensor<T>& HardShrink<T>::updateGradInput(Tensor<T>& input){
    
    input.map2(output,grad);
    return input;
    
  };
  
};


