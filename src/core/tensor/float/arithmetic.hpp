

template<typename T>
constexpr void Tensor<T>::add(const Tensor<T> &t){
  
  //static_assert(this->shape.elements == t.shape.elements,"number of elements have to be same");
  
  //std::cout << t.shape.elements << std::endl;
  
  assert(this->shape.elements == t.shape.elements);
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] += t.data[i];
  }
  
  // this->print();
  
};

template<typename T>
constexpr void Tensor<T>::operator+(const Tensor<T> &t){

  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] += t.data[i];
  }
  
};

template<typename T>
constexpr void Tensor<T>::sub(const Tensor<T> &t){
  
  assert(this->shape.elements == t.shape.elements);
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] -= t.data[i];
  }
  
};

template<typename T>
constexpr void Tensor<T>::operator-(const Tensor<T> &t){
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] -= t.data[i];
  }
};


template<typename T>
constexpr void Tensor<T>::mul(const Tensor<T> &t){

  assert(this->shape.elements == t.shape.elements);
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] *= t.data[i];
  }
  //this->print();  
};

template<typename T>
constexpr void Tensor<T>::operator*(const Tensor<T> &t){
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] *= t.data[i];
  }
};


template<typename T>
constexpr void Tensor<T>::div(const Tensor<T> &t){
  
  assert(this->shape.elements == t.shape.elements);
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] /= t.data[i];
  }
  //this->print();
};

template<typename T>
constexpr void Tensor<T>::operator/(const Tensor<T> &t){
  
  for(size_t i = 0 ; i < this->shape.elements ; i++){
    this->data[i] /= t.data[i];
  }
  
};


template<typename T>
constexpr void Tensor<T>::dot(const Tensor<T> &t,const Tensor<T> &out,std::string type){
  
  /*
    type specifies either rows or columns of first and second matrix which attributes dimensions of output matrix.
    For instance , type "RC" denotes number of rows of first matrix and columns of second matrix is going to be output dimensions.
    There are 4 distinct cases for output matrix, namely type = ["RR","RC","CR","CC"]. The mulitiplicity enables us to operate 
    the computations of neural network without transpositions of matrixes.
  */
  
  size_t d = ((type == "RC") || (type == "CC"))? t.shape.dimensions[0] : t.shape.dimensions[1];
  size_t outC = ((type == "RC") || (type == "CC"))? t.shape.dimensions[1] : t.shape.dimensions[0];
  size_t outR = ((type == "RR") || (type == "RC"))? this->shape.dimensions[0] : this->shape.dimensions[1];
  
  // std::cout << t.shape.dimensions[1] << "," << t.shape.dimensions[0] << "," << this->shape.dimensions[1] << std::endl;
  // std::cout << d << "," << outC << "," << outR << std::endl;
  
  T temp = 0;
  
  //#ifdef _OPENMP
  //#pragma omp parallel for
  //#endif
  
  for(size_t i = 0 ; i < outC ; i++ )
    for(size_t j = 0 ; j < outR ; j++ ){   
      for(size_t k = 0 ; k < d ; k++ )
	temp += this->data[j*d+k] * t.data[i*d+k];
      out.data[i*outR+j] = temp;temp = 0;
    }  
  
};

