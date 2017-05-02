
#include "module.hpp"

namespace nn {
  template<typename T>
  class Sigmoid : public nn::Module {
    
    Tensor<T> output;
    const std::function<T(T)> operation = [](T x ) -> T {return 1 / ( 1 + exp(-x) );};
    const std::function<T(T,T)> grad = [](T x,T y) -> T {return x * ( y * ( 1 - y ));};
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    void accGradParameters();
    void name() const;
    
  protected:
    
  public:
    
    Sigmoid();
    
    
    
  };
  
  template<typename T> Sigmoid<T>::Sigmoid(){};
  
  template<typename T> Tensor<T>& Sigmoid<T>::updateOutput(Tensor<T>& input){
    
    input.apply(operation);

    output.resize(input.shape.dimensions);
    input.copy(output);
    
    return input;
  };
  
  template<typename T> Tensor<T>& Sigmoid<T>::updateGradInput(Tensor<T>& input){
    
    input.map2(output,grad);
    
    return input;
  };
  
  template<typename T>
  void Sigmoid<T>::accGradParameters(){
    //std::cout << "nothing is going to be updated in this layer..\n";
  };
  
  template<typename T>
  void Sigmoid<T>::name() const{
    std::cout << "Sigmoid" << std::endl;
  };
  
};
