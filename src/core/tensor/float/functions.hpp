


template<typename T>
constexpr void Tensor<T>::apply(const std::function<T(T)> f){

  //#ifdef _OPENMP
  //#pragma omp parallel for
  //#endif
  
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    this->data[i] = f(data[i]); 
  }
  
  //std::cout << typeid(T).name() << std::endl;  
};

template<typename T>
inline constexpr void Tensor<T>::map(const std::function<T(T)> f){  
  return apply(f);
};


template<typename T>
constexpr void Tensor<T>::map2(Tensor<T>& t2,const std::function<T(T,T)> f){
  
  //#ifdef _OPENMP
  //#pragma omp parallel for
  //#endif
  
  for(size_t i = 0 ; i < this->shape.elements ; i++ ){
    this->data[i] = f(data[i],t2.data[i]); 
  }  
};





