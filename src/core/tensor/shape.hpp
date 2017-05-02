
#include <iostream>
#include <memory>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <functional>
#include <limits>
#include <typeinfo>
#include <assert.h>
#include <omp.h>


class Shape{

private:
  
  size_t extraBits;
  size_t rIndex;
  size_t nBits;
  
  
protected:
  
  
  
public:
  
  Shape(){};
  Shape(const std::vector<size_t> dims);
  Shape(const std::vector<size_t> dims,const std::string type);
  
  
  //Shape(const std::vector<size_t> dims,size_t index,size_t block,size_t nb);
  
  size_t elements;
  size_t typeElements;
  
  std::vector<size_t> dimensions;
  std::vector<size_t> typeDimensions;
  
  int operator[](int n);
  void resize(const std::vector<size_t> &dims);
  
  //std::vector<int> dimensions;
  
  int R();
  
};

Shape::Shape(const std::vector<size_t> dims){
  
  if(dimensions.size() > 0){  
    dimensions.clear();
  }
  
  for(size_t i = 0 ; i < dims.size() ; i++) {
    dimensions.push_back(dims[i]);  
  }
  
  elements = std::accumulate(dimensions.begin(),dimensions.end(),1,std::multiplies<size_t>());
  
};

Shape::Shape(const std::vector<size_t> dims,std::string type)
{
  
  for(size_t i = 0 ; i < dims.size() ; i++) {
    dimensions.push_back(dims[i]);
  }
  
  elements = std::accumulate(dimensions.begin(),dimensions.end(),1,std::multiplies<size_t>());
  
  if(type == "uint32_t"){
    
    typeElements = ( elements + sizeof(uint32_t) * 8 ) / ( sizeof(uint32_t) * 8 ) ;
    
    
    // for(size_t i = 0 ; i < type_elements ; i++){
    //   typeDimensions.push_back
    // }
    
    
  }
  
  //if(T == float)
  //  std::cout << "" << std::endl;
  
  // std::cout << "block : " << nwords << std::endl;
  // std::cout << dims[1] << " " << nwords  << std::endl;
  // std::cout << dims[0] << " : " << dims[index] / nwords + 1 << std::endl;
  // std::cout << dims.size() << std::endl;
  
  for(size_t i = 0 ; i < dims.size() ; i++) {
    
    // if(i == index){
    //   dimensions.push_back(dims[i]/nwords);
    //   dimensions[i] = (dims[index]%nwords == 0) ? dimensions[i] : dimensions[i] + 1;
    //   extraBits = (dims[index]%nwords == 0) ? 0 : nwords - dims[index] % nwords;
    
    // } else {
    //   dimensions.push_back(dims[i]);
    // }
    
    
    
  }
  
  
  
};

void Shape::resize(const std::vector<size_t> &dims){
  
  if(dimensions.size() > 0){
    dimensions.clear();
  }
  
  for(size_t i = 0 ; i < dims.size() ; i++ ){
    dimensions.push_back(dims[i]);  
  }
  
  //elements = 10;
  
  elements = std::accumulate(dimensions.begin(),dimensions.end(),1,std::multiplies<int>());
  
  
};

int Shape::operator[](int n) { return *std::next(dimensions.begin(), n);}

int Shape::R(){
  return dimensions.size();
};


