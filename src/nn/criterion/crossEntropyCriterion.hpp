
#include "criterion.hpp"

namespace nn {

  template<typename T>
  class CrossEntropyCriterion{
    
    float error;
    Tensor<T> output;
    Tensor<T> gradInput;
    
  public:
    
    explicit CrossEntropyCriterion(Tensor<T>& weights);
    T updateOutput(Tensor<T>& input,Tensor<T>& target);
    Tensor<T>& updateGradInput();
    
  };
  
  template<typename T>
  CrossEntropyCriterion<T>::CrossEntropyCriterion(Tensor<T>& weights){ 
    
  };
  
  template<typename T>
  T CrossEntropyCriterion<T>::updateOutput(Tensor<T>& input, Tensor<T>& target){
    
    /*
     loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                    = -x[class] + log(\sum_j exp(x[j]))
    */
    
    output.resize(input.shape.dimensions);
    input.copy(output);
    
    output.exp();
    T p = std::log(output.sum());
    std::function<T(T,T)> operation = [&p](T x,T y) -> T {return -y + p;};
    
    output.map2(target,operation);
    
    //output.pow(2);
    error = output.mean();
    
    return error;
    
  };
  
  template<typename T> Tensor<T>& CrossEntropyCriterion<T>::updateGradInput(){
    
    gradInput.resize(output.shape.dimensions);
    gradInput.fill(error / gradInput.shape.elements);
    gradInput.print();
    
    return gradInput;
  };
  
};

