#pragma once

#include "NDArray.h"
#include "String.h"
#include <map>
#include "KoAdd.h"
#include "KoArgMax.h"
#include "KoCat.h"
#include "KoCrossEntropy.h"
#include "KoDiv.h"
#include "KoDot.h"
#include "KoDropout.h"
#include "KoExp.h"
#include "KoGather.h"
#include "KoIndexSelect.h"
#include "KoLog.h"
#include "KoMaskedFill.h"
#include "KoMax.h"
#include "KoMean.h"
#include "KoMul.h"
#include "KoNeg.h"
#include "KoPow.h"
#include "KoRelu.h"
#include "KoRepeat.h"
#include "KoReshape.h"
#include "KoSlice.h"
#include "KoSqueeze.h"
#include "KoSub.h"
#include "KoSum.h"
#include "KoTanh.h"
#include "KoTranspose.h"
#include "KoUnsqueeze.h"


typedef std::shared_ptr<class Tensor> TensorPtr;
class Tensor : public std::enable_shared_from_this<Tensor>
{
	static inline int		_nextId = 0;

	const int				_id;
	std::vector<TensorPtr>	_creators;		// Tensors used to create this tensor (probably should be encapsulated in KernelOp).
	std::map<size_t,int>	_children;		// Tensors derived from this tensor (all need to be processed before this can backprop).
	KernelPtr				_kernel;		// Kernal operation used to produce the data.
	const bool				_autograd;		// Enable backprop on the and derived.

	NDArray					_data;			// Data array.
	
	TensorPtr				_gradient;		// Gradient of loss with respect to this.
	
	TensorPtr				_momentum;		// Optimizer, momentum.
	TensorPtr				_momentum2;		// Optimizer, second momentum (ADAM).

	std::string				_name;			// Optional name available to model.

	static inline int NextId()
	{
		return ++_nextId;
	}

	bool AllChildrenGradsAccountedFor() const
	{
		for(auto& c:_children)
			if(c.second>0)
				return false;
		return true;
	}

	Tensor(const Tensor&) = delete;
	void operator = (const Tensor&) = delete;

	static NDArrays Arrays(const std::vector<TensorPtr>& tensors)
	{
		NDArrays arrays;
		arrays.reserve(tensors.size());
		for(auto& tensor:tensors)
			arrays.emplace_back(tensor->_data);
		return arrays;
	}

	void AddAsChild() const
	{
		// Add this as a child of it's creators.
		//std::cout<<"[";
		//bool first = true;
		for(auto& c:_creators)
		{
//			if(first)
	//			first = false;
		//	else
			//	std::cout<<",";
			//std::cout<<c->_id;
			if(c->_children.find(_id)==c->_children.end())
				c->_children[_id] = 1;
			else
				c->_children[_id]++;
		}
		//std::cout<<"]="<<_id<<std::endl;
	}

	class P
	{
	};

public:
	template<class iter>
	Tensor(const P&,const NDArray& data,bool autograd,const iter& begin,const iter& end) :
		_id(NextId()),
		_autograd(autograd),
		_data(data)
	{
		// Store list of creators.
		for(auto i=begin;i!=end;++i)
			_creators.emplace_back(*i);

		// Add this as a child of it's creators.
		AddAsChild();
	}

	template<class iter>
	Tensor(const P&,const bool autograd,const iter& begin,const iter& end,const KernelPtr& kernel) :
		_id(NextId()),
		_kernel(kernel),
		_autograd(autograd),
		_creators(begin,end),
		_data(kernel->Forward(Arrays(_creators)))
	{
		// Add this as a child of it's creators.
		AddAsChild();
	}

	// Added for CategoricalDistribution, need to create a tensor derived from a shape vector.
	Tensor(const P&,const NDShape& shape) :
		_data(NDData::New(shape)),
		_autograd(false),
		_id(NextId())
	{
	}

	static TensorPtr New(const NDArray& data,bool autograd=false,const std::initializer_list<TensorPtr>& creators={})
	{
		return std::make_shared<Tensor>(P(),data,autograd,creators.begin(),creators.end());
	}

	static TensorPtr New(const bool autograd,const std::vector<TensorPtr>& creators,const KernelPtr& creationOp)
	{
		return std::make_shared<Tensor>(P(),autograd,creators.begin(),creators.end(),creationOp);
	}

	// Added for CategoricalDistribution, need to create a tensor derived from a user defined shape.
	static TensorPtr New(const NDShape& shape)
	{
		return std::make_shared<Tensor>(P(),shape);
	}

	static TensorPtr New(const std::initializer_list<int>& shape,const std::vector<FP>& data)
	{
		return Tensor::New(NDData::New(shape,data),false,{});
	}

	const std::string& Name() const
	{
		return _name;
	}

	void Name(const std::string& name)
	{
		_name = name;
	}

	TensorPtr Add(const TensorPtr& other) const
	{
		return Tensor::New(_autograd,{Self(),other},std::make_shared<KoAdd>());
	}

	TensorPtr ArgMax(const int dim) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoArgMax>(dim));
	}

	void AssertReadyForBackprop(int nest=0) const
	{
		for(int i=0;i<nest;i++)
			std::cout << ".";
		if(!_gradient)
			throw "No gradient";
		for(auto& child:_children)
		{
			if(child.second!=0)
			{
				std::cout<<"Tensor "<<std::to_string(_id)<<" missing contribution from child "<<std::to_string(child.first)<<"."<<std::endl;
				throw "Child not processed!";
			}
		}
		for(auto& creator:_creators)
			creator->AssertReadyForBackprop(nest+1);
	}

	TensorPtr ReverseBroadcast(const TensorPtr& creator)
	{
		if(Shape()==creator->Shape())
			return Self();
		else
			return Tensor::New(NDData::ReverseBroadcast(_data,creator->Shape()));
	}

	void Backward(const TensorPtr& gradient,const TensorPtr& child)
	{
		if(!child&&_children.size()>0)
			throw ChildNotSpecified();
		if(child&&_children.size()==0)
			throw ChildNotSpecified();

		//gradient->Print();

		//std::cout<<_id<<std::endl;
		//if(_id==412)
//			int a = 1;

		if(_autograd)
		{
			// Acknowledge child contribution to gradient.
			if(child)
			{
				const auto i = _children.find(child->_id);
				if(i==_children.end())
					throw MissingChild();
				if(i->second==0)
					throw AlreadyPropagated();
				else
					i->second--;
			}

			if(!_gradient)
			{
				// First/only gradient passed back.
				_gradient = gradient;
				_momentum = Tensor::New(_gradient->_data.Zeros());
				_momentum2 = Tensor::New(_gradient->_data.Zeros());
			}
			else
			{
				// Subsequent gradient passed back, add to existing gradient.
				// Inplace addition is more efficient and also creating a new Tenosr can create a circular reference.
				_gradient->_data += gradient->_data;
			}

			// Shape of gradient should be the same as (this) the Tensor it relates to - there should be a gradient for each value.
			if(_data.Shape()!=_gradient->_data.Shape())
				throw IncompatibleShape(_data.Shape(),_gradient->_data.Shape());

			// Backprop to tensor(s) used to create this tensor if gradient is ready.
			if(_creators.size()>0&&AllChildrenGradsAccountedFor())
			{
				NDArrays inputs;
				for(auto& creator:_creators)
					inputs.emplace_back(creator->_data);
				const NDArrays gradients = _kernel->Backward(_gradient->_data,inputs);
				for(size_t i=0;i<_creators.size();++i)
					_creators[i]->Backward(Tensor::New(gradients[i]),Self());
			}
		}
	}

	void Backward()
	{
		// Assume default gradient (i.e. first backward pass, gradient WRT itself is 1).
		Backward(Tensor::New(_data.Ones()),nullptr);
	}


	// Concatenates tensors along dimension.
	//
	static TensorPtr Cat(const std::vector<TensorPtr>& tensors,int dim)
	{
		// Reverse dim here to simplify backprop.
		if(dim<0)
			dim += tensors[0]->Shape().size();

		// Check need for gradient.
		bool autograd = false;
		for(auto& tensor:tensors)
			autograd |= tensor->_autograd;

		return Tensor::New(autograd,tensors,std::make_shared<KoCat>(dim));
	}


	TensorPtr CrossEntropy(const TensorPtr& targets) const
	{
		// Must be one target 'index' for each sample in the batch.
		if(targets->Shape().size()!=1||targets->Shape()[0]!=Shape()[0])
			throw IncompatibleShape();

		return Tensor::New(_autograd,{Self()},std::make_shared<KoCrossEntropy>(targets->_data));
	}

	TensorPtr Div(const TensorPtr& other) const
	{
		return Tensor::New(_autograd,{Self(),other},std::make_shared<KoDiv>());
	}

	TensorPtr Dot(const TensorPtr& other) const
	{
		return Tensor::New(_autograd,{Self(),other},std::make_shared<KoDot>());
	}

	TensorPtr Dropout(const FP p) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoDropout>(p));
	}

	TensorPtr Equal(const TensorPtr& other) const
	{
		if(_autograd)
			throw NotImplemented();
		else
			return Tensor::New(_data==other->_data);
	}

	TensorPtr Exp() const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoExp>());
	}

	// Repeat n times - note this "projects" each row, and does not simply stack n copies of the data.
	//
	TensorPtr Repeat(const int dim,const int copies) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoRepeat>(dim,copies));
	}

	TensorPtr Gather(const int dim,const TensorPtr& indices) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoGather>(dim,indices->_data));
	}
	
	const TensorPtr& Gradient() const
	{
		return _gradient;
	}

	const FP Item() const
	{
		return _data[{}];
	}

	const TensorPtr& Momentum() const
	{
		return _momentum;
	}

	const TensorPtr& Momentum2() const
	{
		return _momentum2;
	}

	TensorPtr IndexSelect(const TensorPtr& indices) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoIndexSelect>(indices->_data));
	}

	bool IsEqualTo(const TensorPtr& v) const
	{
		return _data.IsEqualTo(v->_data);
	}

	static TensorPtr Load(const std::string& filename,const bool autograd=false)
	{
		return Tensor::New(NDData::Load(filename),autograd);
	}

	TensorPtr Log() const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoLog>());
	}

	TensorPtr MaskedFill(const TensorPtr& mask,const FP value)
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoMaskedFill>(mask->_data,value));
	}

	TensorPtr Max(int dim) const
	{
		// Reverse dim here to simplify backprop.
		if(dim<0)
			dim += (int)Shape().size();

		return Tensor::New(_autograd,{Self()},std::make_shared<KoMax>(dim));
	}

	TensorPtr Mean(const bool keepDims) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoMean>(keepDims));
	}

	TensorPtr Mean(const int dim,const bool keepDims) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoMean>(dim,keepDims));
	}

	TensorPtr Mul(const TensorPtr& other) const
	{
		return Tensor::New(_autograd,{Self(),other},std::make_shared<KoMul>());
	}

	TensorPtr Neg() const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoNeg>());
	}

	TensorPtr Pow(const FP exponent) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoPow>(exponent));
	}

	TensorPtr Relu() const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoRelu>());
	}

	TensorPtr Reshape(const std::initializer_list<int>& shape) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoReshape>(shape));
	}

	void Save(const std::string& filename) const
	{
		_data.Save(filename);
	}

	const NDShape& Shape() const
	{
		return _data.Shape();
	}

	TensorPtr Slice(const std::initializer_list<std::initializer_list<int>>& slicer)
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoSlice>(slicer));
	}

	TensorPtr Softmax(const int dim) const
	{
		// KeepDim=true else backprop would need to add it back (old comment,feel free to change this).
		// This likely would be more efficient not use add this to the tensor graph and instead add analytic backprop for softmax.
		// Here we use the numerically equivalent softmax where max(x) is subtracted to avoid e^(large) overflowing.
		// This avoids overflows which would be catastrophic and instead e^(-large) is close to zero and so does not affect the result.
		// The word 'large' here is just to indtcate magnitude and does not imply max(x) has it's sign flipped.
		const TensorPtr expo = Sub(Max(dim))->Exp();
		return expo->Div(expo->Sum(dim,true));
	}

	TensorPtr Sqrt() const
	{
		return Pow(0.5);
	}

	TensorPtr Squeeze(const int dim) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoSqueeze>(dim));
	}

	// Stack row vectors into a matrix.
	//
	static TensorPtr Stack(const std::vector<TensorPtr>& tensors)
	{
		// Each row vector must be the same length.
		const int length = tensors[0]->Shape()[0];
		
		// Create matrix.
		const TensorPtr r = Tensor::New(NDData::New({(int)tensors.size(),length}));

		for(int i=0;i<tensors.size();++i)
		{
			const TensorPtr tensor = tensors[i];

			if(tensor->Shape().size()!=1||tensor->Shape()[0]!=length)
				throw IncompatibleShape();
			
			r->_data.Slice({{i}}) = tensor->_data;
		}

		return r;
	}

	TensorPtr Sub(const TensorPtr& other) const
	{
		return Tensor::New(_autograd,{Self(),other},std::make_shared<KoSub>());
	}

	TensorPtr Sum(int dim,const bool keepDims) const
	{
		// Reverse dim here to simplify backprop.
		if(dim<0)
			dim += (int)Shape().size();

		return Tensor::New(_autograd,{Self()},std::make_shared<KoSum>(dim,keepDims));
	}

	TensorPtr Tanh() const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoTanh>());
	}

	TensorPtr Transpose() const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoTranspose>());
	}

	TensorPtr Tril() const
	{
		if(_autograd)
			throw NotImplemented();
		else
			return Tensor::New(_data.Tril());
	}

	TensorPtr Unsqueeze(const int dim) const
	{
		return Tensor::New(_autograd,{Self()},std::make_shared<KoUnsqueeze>(dim));
	}

	TensorPtr Var() const
	{
		if(_autograd)
			throw NotImplemented();
		else
			return Tensor::New(_data.Var());
	}

	TensorPtr Var(const int dim,const bool keepDim,const int correction=1/*Bessel's correction*/) const
	{
		// Implemented here so backprop is free.
		TensorPtr r = Sub(Mean(dim,true));
		r = r->Pow(2.0);
		r = r->Sum(dim,keepDim);
		r = r->Div(Tensor::New(NDData::New({},Shape()[dim]-FP(1.0))));
		return r;
	}

	TensorPtr Self() const
	{
		return std::const_pointer_cast<Tensor>(shared_from_this());
	}

	const NDArray& Data() const
	{
		return _data;
	}

	NDArray& Data()
	{
		return _data;
	}

	void ClearChildren()
	{
		_children.clear();
	}

	void Print(std::ostream& out=std::cout) const
	{
		_data.Print(out);
	}

	static TensorPtr Ones(const std::initializer_list<int>& shape)
	{
		return New(NDData::New(shape,1.0));
	}

	static TensorPtr Zeros(const std::initializer_list<int>& shape)
	{
		return New(NDData::New(shape,0.0));
	}
};