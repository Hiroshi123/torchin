

template<>
void Tensor<uint32_t>::operator & (const Tensor<uint32_t>& t2){
  
  //static_assert(t2.shape.typeElements == 1,"");
  //static_assert(this->shape.typeElements == t2.shape.typeElements,"");
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    this->data[i] &= t2.data[i];
  }
  
};

template<>
void Tensor<uint32_t>::operator | (const Tensor<uint32_t>& t2){
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    this->data[i] |= t2.data[i];
  }
};

template<>
void Tensor<uint32_t>::operator << (const size_t s){
  
  uint32_t shif;
  for(int i = this->shape.typeElements - 1 ; i >= 0 ; i--){
    shif = this->data[i] & ((1 << (32 - s)) - 0);
    this->data[i] >>= s;
    if(i != (int) this->shape.typeElements - 1){
      //std::cout << "dd::" << shif << " : " << (shif << (sizeof(uint32_t) * 8 - 1)) << std::endl;
      this->data[i] += shif; //shif << (sizeof(uint32_t) * 8 - 2);
    }
  }
  
};

template<>
void Tensor<uint32_t>::operator >> (const size_t s){
  
  uint32_t shif;
  for(int i = this->shape.typeElements - 1 ; i >= 0 ; i--){
    
    //std::cout << "data " << this->data[i] << "\n";
    //shif = this->data[i] & ((1 << s) - 1);
    
    shif = this->data[i] & ((1 << (32 - s)) - 1);
    
    this->data[i] <<= s;
    
    // std::cout << "shif " << shif << std::endl;
    
    // std::cout << "dd" << (shif >> (sizeof(uint32_t) * 8 - 2)) << std::endl;
    
    //this->data[i+1] += shif;
    
    //this->data[i+1] += shif << (sizeof(uint32_t) * 8 - 1) ;
    
    if(i != (int) this->shape.typeElements - 1){
      //std::cout << shif << (sizeof(uint32_t) * 8 - 1) << "\n";
      this->data[i+1] += shif >> (sizeof(uint32_t) * 8 - 2);
      //this->data[i+1] += shif << (sizeof(uint32_t) * 8 - 1) ;
    }
  }
  
};


// template<>
// void Tensor<uint32_t>::operator ~ (const Tensor<uint32_t>& t2){
//   for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
//     this->data[i] ^= t2.data[i];
//   }  
// };


template<>
void Tensor<uint32_t>::operator ^ (const Tensor<uint32_t>& t2){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    this->data[i] ^= t2.data[i];
  }
  
};

template<>
void Tensor<uint32_t>::OR (const Tensor<uint32_t>& t2,const Tensor<uint32_t>& out){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    out.data[i] = this->data[i] | t2.data[i];
  }  
};

template<>
void Tensor<uint32_t>::AND (const Tensor<uint32_t>& t2,const Tensor<uint32_t>& out){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    out.data[i] = this->data[i] & t2.data[i];
  }
};

template<>
void Tensor<uint32_t>::XOR (const Tensor<uint32_t>& t2,const Tensor<uint32_t>& out){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    out.data[i] = this->data[i] ^ t2.data[i];
  }
};

template<>
void Tensor<uint32_t>::FLIP (){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    this->data[i] = ~ this->data[i];
  }
  
};

template<>
void Tensor<uint32_t>::FLIP (const Tensor<uint32_t>& out){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){  
    out.data[i] = ~ this->data[i];
  }
};

template<>
void Tensor<uint32_t>::RSHIFT(const size_t s,const Tensor<uint32_t>& out){
  
  for(size_t i = 0 ; i < this->shape.typeElements ; i++){
    uint32_t shif = this->data[i] & 1 ;
    this->data[i] <<= 1;
    this->data[i] += shif << (sizeof(uint32_t) * 8 - 1) ;
  }
  
};

template<>
void Tensor<uint32_t>::LSHIFT(const size_t s,const Tensor<uint32_t>& out){
  
};


