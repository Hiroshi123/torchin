
#include "module.hpp"

namespace nn {

  template<typename T>
  class Relu : public nn::Module {

  private:
    
    Tensor<T>& updateOutput(Tensor<T>&);
    Tensor<T>& updateGradInput(Tensor<T>&);
    void name() const;
    const std::function<T(T)> operation = [](T x) -> T {return x > 0 ? x : 0;};
    
  public:

    Relu();
    
    //Tensor<>* t;
    
    
    //Tensor<>* updateOutput();
    
    
    
  };
  
  template<typename T> Relu<T>::Relu(){
    
    //t->resize({1,1});
    
  };
  
  template<typename T>
  Tensor<T>& Relu<T>::updateOutput(Tensor<T>& input){  
    input.apply(operation);
    return input;
  };
  
  template<typename T>
  Tensor<T>& Relu<T>::updateGradInput(Tensor<T>& input){
    return input;    
  };
  
  template<typename T>
  void Relu<T>::name() const {
    std::cout << "rectified linear unit" << std::endl;
  };
  
};
