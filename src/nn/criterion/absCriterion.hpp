
#include "criterion.hpp"

namespace nn {

  template<typename T>
  class absCriterion{
    
    float error;
    Tensor<T> output;
    Tensor<T> gradInput;
    
  public:
    
    explicit absCriterion() = default;
    T updateOutput(Tensor<T>& input,Tensor<T>& target);
    Tensor<T>& updateGradInput();
    
  };
  
  template<typename T>
  T absCriterion<T>::updateOutput(Tensor<T>& input, Tensor<T>& target){
    
    /*
     loss(x, y)  = 1/n \sum |x_i - y_i|
    */
    
    output.resize(input.shape.dimensions);
    input.copy(output);
    output.sub(target);
    //output.pow(2);
    error = output.mean();
    
    return error;
    
  };
  
  template<typename T> Tensor<T>& absCriterion<T>::updateGradInput(){
    
    gradInput.resize(output.shape.dimensions);
    gradInput.fill(error / gradInput.shape.elements);
    gradInput.print();
    
    return gradInput;
  };
  
};
