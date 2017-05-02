

template<>
void Tensor<uint32_t>::print() {
  
  for(size_t j = 0 ; j < this->shape.typeElements ; j++ ){
    
    for(size_t i = 0 ; i < 32 ; i++){
      
      ( this->data[j] & (1 << i) ) ? std::cout << "1," : std::cout << "0,";
      
      if( (this->shape.R() == 2) & ( ( j * 32 + i + 1 ) % this->shape.dimensions[1] == 0 ) ){
	std::cout << "\n";
      }
      
      //(j * 32 + i == this->shape.elements - 1 ) ? break() : {} ;
      
      if(j * 32 + i == this->shape.elements - 1 )
	break;
      
    }
    
  }
  
  size_t r = this->shape.R();
  
  std::cout << std::endl;
  
  std::cout << "[Tensor(";
  for(size_t i = 0 ; i < r ; i++){
    if(i != r-1)
      std::cout << this->get_shape().dimensions[i] << "x";
    else
      std::cout << this->get_shape().dimensions[i];
  }
  
  std::cout << ")]" << std::endl;
  std::cout << std::endl;
  
};

