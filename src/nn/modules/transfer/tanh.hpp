
#include "module.hpp"

namespace nn{
  
  template<typename T>
  class Tanh : public nn::Module {

    Tensor<T> output;
    std::function<T(T)> operation = [](T x) -> T {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));};
    std::function<T(T,T)> grad = [](T x,T y) -> T {return x * ( - y * y + 1 );};
    
  protected:
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    
    //void accGradParameters();
    
  public:
    
    Tanh() = default;
    
  };
  
  //template<typename T> Tanh<T>::Tanh(){};
  
  template<typename T>
  Tensor<T>& Tanh<T>::updateOutput(Tensor<T>& input){
    
    input.apply(operation);
    output.resize(input.shape.dimensions);
    input.copy(output);
    return input;
    
  };

  template<typename T>
  Tensor<T>& Tanh<T>::updateGradInput(Tensor<T>& input){
    
    input.map2(output,grad);
    return input;
    
  };
  
};
