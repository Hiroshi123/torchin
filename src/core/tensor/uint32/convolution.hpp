

template<>
void Tensor<uint32_t>::convolution1D(const Tensor<uint32_t>& filter,const Tensor<uint32_t>& out,const size_t padding,const size_t stride){
  
  assert(this->shape.R() == 1);
  //assert(filter.shape.R() == 1);
  
  Tensor<uint32_t> buffer(this->shape.dimensions);
  
  size_t middle = ((filter.shape.typeElements - 1) * 32 + filter.shape.elements ) / 2 ;
  
  int rshiftV,lshiftV = 0;
  
  for(size_t j = 0 ; j < filter.shape.typeElements ; j++ )
    for(size_t i = 0 ; i < filter.shape.elements ; i++){
      
      rshiftV = ( (int)(i-middle) > 0 )? (int)(i-middle) : 0;
      lshiftV = ( (int)(middle-i) > 0 )? (int)(middle-i) : 0;
      
      (filter.data[j] & (1 << i) ) ? this->FLIP(buffer) : this->copy(buffer);
      
      buffer >> rshiftV;
      buffer << lshiftV;
      
      buffer.print();
      
      //buffer.unset();
      
      //out.data[0] = (filter.data[j] & (1 << i) )? ((~this->data[0] << lshiftV) << rshiftV) : ((this->data[0] << lshiftV) << rshiftV);
      
    }
  
  
  //std::cout << filter.data[0] << std::endl;
  
  
  //size_t inputD  = this->shape.dimensions[0];
  //size_t kernelW = filter.shape.dimensions[0];
  
  //size_t pad = (padding == std::numeric_limits<size_t>::max() ) ? kernelW / 2 : padding;
  
  //std::cout << std::numeric_limits<size_t>::max() << std::endl;
  
  
  //std::cout << "pad : " << pad << std::endl;
  
  
  // size_t stock = 0;
  // for(size_t i = 0 ; i < inputD ; i++){
  //   for(size_t f = 0 ; f < kernelW ; f++ ){
  //     stock += this->data[i-pad+f] * filter.data[f];
  //   }
  //   out.data[i] = stock ; stock = 0; 
  // }
  
};


template<>
void Tensor<uint32_t>::convolution2D(const Tensor<uint32_t>& filter,const Tensor<uint32_t>& out,
				     const size_t paddingW,const size_t paddingH,
				     const size_t strideW,const size_t strideH
				     ){
  
  /*
    following is an implementation which is I name "shift based convolution"
  */
  
  size_t resW = this->shape.dimensions[0];
  
  //size_t resH = this->shape.dimensions[1];
  
  Tensor<uint32_t> buffer(this->shape.dimensions);
  
  size_t middleW = this->shape.dimensions[0];
  
  //size_t middleH = this->shape.dimensions[1];
  
  //((filter.shape.typeElements - 1) * 32 + filter.shape.elements ) / 2 ;
  
  int rshiftV = 0 ,lshiftV = 0 ,upshiftV = 0 ,downshiftV = 0;
  
  for(size_t j = 0 ; j < filter.shape.typeElements ; j++ )
    for(size_t i = 0 ; i < filter.shape.elements ; i++){
      
      rshiftV = ( (int)(i-middleW) > 0 )? (int)(i-middleW) : 0;
      lshiftV = ( (int)(middleW-i) > 0 )? (int)(middleW-i) : 0;
      
      (filter.data[j] & (1 << i) ) ? this->FLIP(buffer) : this->copy(buffer);
      
      buffer << ( 1 << ( upshiftV   * resW ) );
      buffer >> ( 1 >> ( downshiftV * resW ) );
      
      buffer >> rshiftV;
      buffer << lshiftV;
      
      buffer.print();
      
      //buffer.unset();
      
      //out.data[0] = (filter.data[j] & (1 << i) )? ((~this->data[0] << lshiftV) << rshiftV) : ((this->data[0] << lshiftV) << rshiftV);
      
    }
  
};


