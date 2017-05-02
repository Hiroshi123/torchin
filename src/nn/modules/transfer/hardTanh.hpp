
#include "module.hpp"


namespace nn {
  template<typename T>
  class HardTanh : public nn::Module {
    
    Tensor<T> output;
    const std::function<T(T)> operation = [](T x ) -> T {return 1 / ( 1 + exp(-x) );};
    const std::function<T(T,T)> grad = [](T x,T y) -> T {return x * ( y / ( 1 - y ));};
    
  protected:
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    void accGradParameters();
    
    
  public:
    
    HardTanh();
    
    void name() const;
    
  };
  
  template<typename T> HardTanh<T>::HardTanh(){};
  
  template<typename T> Tensor<T>& HardTanh<T>::updateOutput(Tensor<T>& input){
    
    output.resize(input.shape.dimensions);
    input.copy(output);
    input.apply(operation);
    
    return input;
  };
  
  template<typename T> Tensor<T>& HardTanh<T>::updateGradInput(Tensor<T>& input){
    
    input.map2(output,grad);
    return input;
  };
  
  template<typename T>
  void HardTanh<T>::accGradParameters(){
    //std::cout << "nothing is going to be updated in this layer..\n";
  };
  
  template<typename T>
  void HardTanh<T>::name() const{
    std::cout << "HardTanh" << std::endl;
  };
  
};




