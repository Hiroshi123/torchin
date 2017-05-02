

template<typename T>
constexpr void Tensor<T>::convolution1D(const Tensor<T>& filter,const Tensor<T>& out,const size_t padding,const size_t stride){
  
  assert(this->shape.R() == 1);
  assert(filter.shape.R() == 1);
  
  size_t inputD  = this->shape.dimensions[0];
  size_t kernelW = filter.shape.dimensions[0];
  
  size_t pad = (padding == std::numeric_limits<size_t>::max() ) ? kernelW / 2 : padding;
  
  std::cout << std::numeric_limits<size_t>::max() << std::endl;
  std::cout << "pad : " << pad << std::endl;
  
  size_t stock = 0;
  for(size_t i = 0 ; i < inputD ; i++){
    for(size_t f = 0 ; f < kernelW ; f++ ){
      stock += this->data[i-pad+f] * filter.data[f];
    }
    out.data[i] = stock ; stock = 0; 
  }
  
};

template<typename T>
constexpr void Tensor<T>::convolution2D(const Tensor<T>& filter,const Tensor<T>& out,
					const size_t paddingW, const size_t paddingH,
					const size_t strideW,const size_t strideH){
  
  assert(this->shape.R() == 2);
  assert(filter.shape.R() == 2);
  
  size_t resW = this->shape.dimensions[0];
  size_t resH = this->shape.dimensions[1];
  
  size_t kernelW = filter.shape.dimensions[0];
  size_t kernelH = filter.shape.dimensions[1];
  
  size_t padW = (paddingW == std::numeric_limits<size_t>::max() ) ? kernelW / 2 : paddingW;
  size_t padH = (paddingH == std::numeric_limits<size_t>::max() ) ? kernelH / 2 : paddingH;
  
  size_t stock = 0;
  for(size_t ih = 0 ; ih < resH ; ih++){
    for(size_t iw = 0 ; iw < resW ; iw++){
      for(size_t kh = 0 ; kh < kernelH ; kh++ ){
	for(size_t kw = 0 ; kw < kernelW ; kw++ ){
	  if( ( iw == 0 && kw == 0 ) || ( ih == 0 && kh == 0) || (iw == resW-1 && kw == kernelW-1) || (ih == resH-1 && kh == kernelH-1) ){
	    /* for padding */
	    //std::cout << iw << "," << ih << "," << kw << "," << kh << std::endl;
	  } else
	    stock += this->data[ ( ih - padH ) * resW + ( iw - padW ) + ( kh * resW + kw ) ] * filter.data [ kh * kernelW + kw ];
	}
      }
      out.data[ iw * resH + ih ] = stock;stock = 0;
    }
  }
  
};

template<typename T>
constexpr void Tensor<T>::convolution3D(const Tensor<T>& filter,const Tensor<T>& out,
					const size_t paddingW, const size_t paddingH,
					const size_t strideW,const size_t strideH){
  
  /*
    assume input tensor is  4D (3D) [ batchsize( or extra dimension ) ,resolution width,resolution height,input feature number ]
           kernel tensor is 4D [ output feature number , kernel width , kernel height , input feature number ]
	   output tensor is 4D (3D) [batchsize (or extra dimension ), resolution width , resolution height , output feature number ]
  */
  
  assert(this->shape.R() == 3 || this->shape.R() == 4);
  assert(filter.shape.R() == 4);
  
  //size_t inputF =  this->shape.dimensions[3];
  
  //size_t resW,resH,nOutputPlane,kW,kH,nInputPlane;
  size_t resW = this->shape.dimensions[0];
  size_t resH = this->shape.dimensions[1];
  size_t nOutputPlane = filter.shape.dimensions[0];
  size_t kW = filter.shape.dimensions[1];
  size_t kH = filter.shape.dimensions[2];
  size_t nInputPlane = filter.shape.dimensions[3];
  
  size_t padW = (paddingW == std::numeric_limits<size_t>::max() ) ? kW / 2 : paddingW;
  size_t padH = (paddingH == std::numeric_limits<size_t>::max() ) ? kH / 2 : paddingH;
  
  size_t stock = 0;
  for(size_t o = 0 ; o < nOutputPlane ; o ++ ){
    for(size_t ih = 0 ; ih < resH ; ih ++ ){
      for(size_t iw = 0 ; iw < resW ; iw ++ ){
  	for (size_t kh = 0 ; kh < kH ; kh ++ ){
  	  for (size_t kw = 0 ; kw < kW ; kw ++ ){
	    for (size_t d = 0 ; d < nInputPlane ; d++ ){
	      if( ( iw == 0 && kw == 0 ) || ( ih == 0 && kh == 0) || (iw == resW-1 && kw == kW-1) || (ih == resH-1 && kh == kH-1) ){
		/* for padding */
		//std::cout << iw << "," << ih << "," << kw << "," << kh << std::endl;
	      } else {
		
		stock +=
		  this->data[ d + kw * nInputPlane * resH + kh * nInputPlane
			      + ( iw - padW ) * nInputPlane * resW + ( ih - padH ) * nInputPlane ]
		  * filter.data[ d + kw * nInputPlane * kH + kh * nInputPlane + (o * kW * kH * nInputPlane)];
		
		//stock += filter.data[ d + kw * nInputPlane + kh * ( nInputPlane * kW) + (o * kW * kH * nInputPlane)];
	      }
	    }
  	  }
	}
  	out.data[ iw * nOutputPlane * resH + ih * nOutputPlane + o ] = stock;stock = 0;
      }
    }
  }
  
};





