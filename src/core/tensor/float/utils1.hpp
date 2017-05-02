

template<typename T>
void Tensor<T>::fill(T x){
  std::fill_n(data.get(), shape.elements, x);
};

template<typename T>
void Tensor<T>::resize(const std::vector<size_t> &dims)
{
  
  shape.resize(dims);
  data.reset(new T[shape.elements]);
  std::fill_n(data.get(), shape.elements, T());
  
  //this->print("dimensions");
  
  //std::cout << "elements " << this->shape.elements << std::endl;
  
  //for(int i = 0 ;  i < 12; i++ )
  //  std::cout << this->data[i] << std::endl;
  
  //int s = this->shape.R();
  //std::cout << s << std::endl;
  
  //size_t r = this->get_shape().R();
  //std::cout << r << std::endl;
  //this->print();
  
};

template<typename T>
void Tensor<T>::reshape(const std::vector<size_t> &v){
  this->shape.dimensions = v;
  //this->print("dimensions");  
};


template<typename T>
constexpr void Tensor<T>::transpose(){
  
  size_t dim1 = this->shape.dimensions[0];
  size_t dim2 = this->shape.dimensions[1];
  
  Tensor<T> temp({dim2,dim1});

  for(size_t i = 0 ; i < dim1 ; i++ ){
    for(size_t j = 0 ; j < dim2 ; j++ ){
      temp[j*dim1+i] = this->data[i*dim2+j];
    }
  }
  
  this->resize({dim2,dim1});
  this->data = std::move(temp.data);
  
};

template<typename T>
constexpr void Tensor<T>::transpose(const Tensor<T>& out){
  
  assert( this->R() == 2 );
  assert( out.R() == 2 );
  
  size_t dim1 = this->shape.dimensions[0];
  size_t dim2 = this->shape.dimensions[1];
  
  for(size_t i = 0 ; i < dim1 ; i++ ){
    for(size_t j = 0 ; j < dim2 ; j++ ){
      out.data[j*dim1+i] = this->data[i*dim2+j];
    }
  }
  
};

template<typename T>
Shape& Tensor<T>::get_shape() {
  return shape; 
};

template<typename T>
void Tensor<T>::copy(const Tensor<T>& out){
  for(size_t i = 0; i < this->shape.elements ; i++)
    out.data[i] = this->data[i];
  
};


