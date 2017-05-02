

template<typename T>
constexpr void Tensor<T>::maxPooling1D(const Tensor<T>& out, const size_t kw,const size_t dw){
  
  size_t dW = (dw == std::numeric_limits<size_t>::max())? kw : dw;
  size_t index = 0;
  T maxV = std::numeric_limits<size_t>::lowest();
  for(size_t i = 0 ; i < this->shape.elements ; i+= dW ){

    for(size_t j = 0 ; j < kw ; j++ ){
      maxV = ( maxV < this->data[i+j] )? this->data[i+j] : maxV ;
    }
    
    out.data[index] = maxV ;
    index++;
    maxV = std::numeric_limits<T>::lowest();
  }
  
};

template<typename T>
constexpr void Tensor<T>::maxPooling2D(const Tensor<T>& out, const size_t kw,const size_t kh,
				       const size_t dw,const size_t dh,
				       const size_t padw,const size_t padh
				       ){
  
  size_t resW = this->shape.dimensions[0];
  size_t resH = this->shape.dimensions[1];
  size_t dW = (dw == std::numeric_limits<size_t>::max())? kw : dw;
  size_t dH = (dh == std::numeric_limits<size_t>::max())? kh : dh;
  size_t index = 0;
  
  T maxV = std::numeric_limits<size_t>::lowest();
  for(size_t ih = 0 ; ih < resH ; ih+= dH ){    
    for(size_t iw = 0 ; iw < resW ; iw+= dW ){
      for(size_t h = 0 ; h < kh ; h++ ){
	for(size_t w = 0 ; w < kw ; w++ ){
	  maxV = ( maxV < this->data[ ih * resW + iw + h * resW + w ] )? this->data[ih * resW + iw + h * resW + w] : maxV ;
	}
      }
      out.data[index++] = maxV ; maxV = std::numeric_limits<size_t>::lowest();
    }
  }
};

template<typename T>
constexpr void Tensor<T>::maxPooling2DwithF(const Tensor<T>& out, const size_t kw,const size_t kh,
					    const size_t dw,const size_t dh,
					    const size_t padw,const size_t padh
					    ){
  
  assert(this->shape.R() == 3 || this->shape.R() == 4);
  
  size_t resW = this->shape.dimensions[0];
  size_t resH = this->shape.dimensions[1];
  size_t inputFN = this->shape.dimensions[2];
  
  size_t dW = (dw == std::numeric_limits<size_t>::max())? kw : dw;
  size_t dH = (dh == std::numeric_limits<size_t>::max())? kh : dh;
  size_t index = 0;
  
  T maxV = std::numeric_limits<size_t>::lowest();
  
  for(size_t f = 0 ; f < inputFN ; f++){
    for(size_t ih = 0 ; ih < resH ; ih+= dH ){    
      for(size_t iw = 0 ; iw < resW ; iw+= dW ){
	for(size_t h = 0 ; h < kh ; h++ ){
	  for(size_t w = 0 ; w < kw ; w++ ){
	    maxV = ( maxV < this->data[ ih * resW * inputFN + iw + h * resW * inputFN + w * inputFN ] )?
	      this->data[ih * resW * inputFN + iw + h * resW * inputFN + w * inputFN ] : maxV ;
	  }
	}
	out.data[ f + index * inputFN ] = maxV ; maxV = std::numeric_limits<size_t>::lowest();
	index ++;
      }
    }
    index = 0;
  }
  
};


