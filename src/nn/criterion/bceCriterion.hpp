
#include "criterion.hpp"

namespace nn{
  
  template<typename T>
  class BCECriterion : public nn::Criterion {

    /*
      Binary Cross Entropy Criterion 
    */
    
    float error;
    Tensor<T> output;
    Tensor<T> gradInput;
    
    const std::function<T(T)> square = [](T x) -> T {return x*x;};
    
    T updateOutput(Tensor<T>&,Tensor<T>&) override final;
    Tensor<T>& updateGradInput() override final;
    
    void name() const;
    
  public:
    
    //nn::Module modu;
    
    BCECriterion();
    
    //std::function<T(T)> operation = [](T x) -> T {return 1 / ( 1 + exp(-x) );};
    
  };
  
  template<typename T> BCECriterion<T>::BCECriterion(){};
  
  template<typename T> T BCECriterion<T>::updateOutput(Tensor<T>& input, Tensor<T>& target){
    
    output.resize({input.shape.dimensions[0],input.shape.dimensions[1]});
    
    // output.print();
    
    /* deep copy */
    input.copy(output);
    
    //std::cout << "output" << std::endl;
    
    output.sub(target);
    output.pow(2);
    error = output.mean();
    
    return error;
  };
  
  template<typename T> Tensor<T>& BCECriterion<T>::updateGradInput(){
    
    gradInput.resize(output.shape.dimensions);
    gradInput.fill(error / gradInput.shape.elements);
    // gradInput.print();
    
    return gradInput;
  };
  
};




