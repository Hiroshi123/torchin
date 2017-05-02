


template<>
void Tensor<float>::print() {
  
  std::vector<size_t> switchElem;
  
  size_t r = this->shape.R();
  size_t stride = 1;
  
  for(int i = r-1 ; i >= 0 ; i -- ){
    
    stride *= this->get_shape().dimensions[i];
    switchElem.push_back(stride-1);
    //std::cout << "ass" << switchElem[0] << std::endl;
    
  }
  
  size_t e = this->get_shape().elements;
  
  std::cout << std::endl;
  std::cout << "(";
  
  std::vector<size_t> digit;
  
  if(r > 1)
    for(size_t i = 0 ; i < r - 2 ; i++){
      digit.push_back(1);
      std::cout << digit[i] << ",";
    }
  
  
  std::reverse(begin(digit), end(digit));
  
  //std::vector<int> dims = this->get_shape().dimensions;
  
  std::cout << ".,.) = " << std::endl;
  
  for(size_t i = 0 ; i < e ; i++) {
    
    
    std::cout << "  " << this->data[i];
    
    if(i % (switchElem[0]+1) == switchElem[0])
      std::cout << std::endl;
   
    bool last = true;

    if(r > 2)
      for(size_t j = 0 ; j < r - 2 ; j++ ){
	if(digit[r-3-j] != this->get_shape().dimensions[j]){
	  //std::cout << digit[j] << ":" << dims[j] << std::endl;
	  last = false;
	  break;
	}
      }
    
    
    if(last){}
    
    //if it is more than two dimensions
    
    else if(i % (switchElem[1]+1) == switchElem[1]){
      
      for(int j = r - 2 ; j > 0; j--){
	
	if(j == 1){
	  digit[0] ++;
	  std::cout << "(";
	  
	  for(int k = r - 3 ; k >= 0 ; k-- ){
	    std::cout << digit[k] << ",";
	  }
	  
	  std::cout << ",.,.) = " << std::endl;
	  break;
	}
	
	else if(i % (switchElem[j]+1) == switchElem[j]){
	  
	  std::cout << "(";
	  for(int k = r - 3 ; k >= 0 ; k-- ){
	    
	    digit[k] = (k == j - 1) ? digit[k]+1 : 1;
	    std::cout << digit[k] << ",";
	  }
	  
	  std::cout << ",.,.) = " << std::endl;
	  break;
	}
      }
    }
    
  }
  
  std::cout << "[Tensor(";
  for(size_t i = 0 ; i < r ; i++){
    if(i != r-1)
      std::cout << this->get_shape().dimensions[i] << "x";
    else
      std::cout << this->get_shape().dimensions[i];
  }
  
  
  std::cout << ")]" << std::endl;
  std::cout << std::endl;
  
}

template<typename T>
constexpr void Tensor<T>::print(std::string arg){
  
  if(arg == "dimensions"){

    std::cout << std::endl;
    size_t r = this->shape.R();
    std::cout << "[Tensor(";
    
    for(size_t i = 0 ; i < r ; i++){
      if(i == this->shape.rIndex && i != r-1)
	std::cout << this->get_shape().dimensions[i] << "(x" << sizeof(T)*8 << ")[=" <<
	  this->get_shape().dimensions[i] * sizeof(T)*8 - this->shape.extraBits << "+" <<
	  this->shape.extraBits << "]x";
      else if(i == this->shape.rIndex)
	std::cout << this->get_shape().dimensions[i] << "(x" << sizeof(T)*8 << ")[=" <<
	  this->get_shape().dimensions[i] * sizeof(T)*8 - this->shape.extraBits << "+" <<
	  this->shape.extraBits << "]";
      else if(i != r-1)
	std::cout << this->get_shape().dimensions[i] << "x";
      else
	std::cout << this->get_shape().dimensions[i];
    }
    
    std::cout << ")]" << std::endl;
    std::cout << std::endl;
    
    // int inc = 0;
    // std::cout << std::endl;
    // std::for_each(this->shape.dimensions.begin(),
    // 		  this->shape.dimensions.end(),
    // 		  [&inc](int t){
    // 		    inc ++;
    // 		    std::cout << "D" << inc << " : " << t << std::endl;
    // 		  });
    // std::cout << std::endl;
    
  }
  
  //std::cout << arg << std::endl;
  
}


