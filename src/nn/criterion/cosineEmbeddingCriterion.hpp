
#include "criterion.hpp"

namespace nn {

  template<typename T>
  class CosineEmbeddingCriterion{
    
    float error;
    Tensor<T> output;
    Tensor<T> gradInput;
    
  public:
    
    CosineEmbeddingCriterion() = default;
    T updateOutput(Tensor<T>& input,Tensor<T>& target);
    Tensor<T>& updateGradInput();
    
    size_t margin = 1;
    
    std::function<T(T,T)> operation = [](T x,T y){ return (y == 1)? 1 - cos(x,y) : max(0,cos(x,y)-1);};
    
  };
  
  template<typename T>
  T CosineEmbeddingCriterion<T>::updateOutput(Tensor<T>& input, Tensor<T>& target){
    
    /*
     x_i,  if y_i ==  1loss(x, y) = 1/n
     max(0, margin - x_i), if y_i == -1
    */
    
    output.resize(input.shape.dimensions);
    input.copy(output);
    map2(output,operation);
    error = output.mean();
    return error;
  };
  
  template<typename T> Tensor<T>& CosineEmbeddingCriterion<T>::updateGradInput(){
    
    gradInput.resize(output.shape.dimensions);
    gradInput.fill(error / gradInput.shape.elements);
    
    return gradInput;
  };
  
};



