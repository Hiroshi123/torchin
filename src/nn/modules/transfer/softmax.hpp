
#include "module.hpp"

namespace nn {

  template<typename T>
  class SoftMax : public nn::Module {
    
  public:
    
    SoftMax();
    Tensor<T>& updateOutput(Tensor<T>&);
    
    //std::function<T(T)> compare = [](T x) -> T {return x > 0 : x : 0;};
    
    std::function<T(T)> exxp = [](T x) -> T {return exp(x);};

    void name();
    
  };
  
  template<typename T>
  SoftMax<T>::SoftMax(){
    
  };
  
  template<typename T>
  Tensor<T>& SoftMax<T>::updateOutput(Tensor<T>& input){
    
    //T max = input.max();
    input.apply(exxp);
    T sum = input.sum();
    auto func = [&sum](T x) -> T {return x / sum;};
    input.apply(func);
    return input;
    
  };
  
  template<typename T>
  void SoftMax<T>::name(){
    std::cout << "SoftMax" << std::endl;
  };
  
};
