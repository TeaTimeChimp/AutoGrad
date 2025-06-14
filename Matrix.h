#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include "Random.h"
#include "NDArray.h"


#define PUSHDOWN


class Matrix
{
	friend class Tensor;

	NDArray	_data;
	int		_startRow;
	int		_endRow;
	int		_startCol;
	int		_endCol;

#pragma region Serialization.

	static void ReadInteger(std::fstream& file,std::stringstream& ss,char& c)
	{
		while(c>='0'&&c<='9')
		{
			ss<<c;
			file>>c;
		}
	}

	static double ReadDouble(std::fstream& file,char& c)
	{
		std::stringstream ss;
		ReadInteger(file,ss,c);
		switch(c)
		{
			case '.':
			{
				ss<<'.';
				file>>c;
				ReadInteger(file,ss,c);
				return atof(ss.str().c_str());
			}
			case ']':
			{
				// End of integer array.
				return atof(ss.str().c_str());
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}
	}

	static std::vector<double> Read1DArray(std::fstream& file,char& c)
	{
		std::vector<double> arr;
		while(c!='\0')
		{
			switch(c)
			{
				case ']':
				{
					file>>c;
					return arr;
				}
				case '-':
				{
					file>>c;
					arr.emplace_back(-(FP)ReadDouble(file,c));
					break;
				}
				case '0':
				case '1':
				case '2':
				case '3':
				{
					arr.emplace_back((FP)ReadDouble(file,c));
					break;
				}
				case ' ':
				case '\n':
				{
					// Value separator.
					file>>c;
					break;
				}
				default:
				{
					throw UnexpectedCharacter();
				}
			}
		}
		return std::vector<double>();
	}

	static std::vector<std::vector<double>> Read2DArray(std::fstream& file,char& c)
	{
		std::vector<std::vector<double>> arr;
		arr.emplace_back(Read1DArray(file,c));
		while(c!='\0')
		{
			switch(c)
			{
				case '[':
				{
					file>>c;
					arr.emplace_back(Read1DArray(file,c));
					break;
				}
				case ']':
				{
					// End of array.
					return arr;
				}
				case '\n':
				case ' ':
				{
					// Array separator.
					file>>c;
					break;
				}
				default:
				{
					throw UnexpectedCharacter();
				}
			}
		}
		return std::vector<std::vector<double>>();
	}

	static std::vector<std::vector<double>> Read1Or2DArray(std::fstream& file,char& c)
	{
		std::vector<std::vector<double>> arr;		
		switch(c)
		{
			case '[':
			{
				file>>c;
				arr = Read2DArray(file,c);
				break;
			}
			case ' ':
			case '0':
			{
				std::vector<double> data = Read1DArray(file,c);
				arr.emplace_back(data);
				break;
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}
		return arr;
	}

#pragma endregion

public:

#pragma region Constructors.

	Matrix(int rows,int cols,const std::vector<double>& data) :
		_data(NDData::New({rows,cols},data)),
		_startRow(0),
		_endRow(rows),
		_startCol(0),
		_endCol(cols)
	{
	}

	Matrix(const Matrix& v) :
		_data(NDData::New(*v._data)),
		_startRow(v._startRow),
		_endRow(v._endRow),
		_startCol(v._startCol),
		_endCol(v._endCol)
	{
	}

	Matrix(int rows,int cols,double value=0.0) :
		_data(NDData::New({rows,cols},value)),
		_startRow(0),
		_endRow(rows),
		_startCol(0),
		_endCol(cols)
	{
	}

	Matrix(int startRow,int endRow,int startCol,int endCol,const NDArray& data) :
		_data(data),
		_startRow(startRow),
		_endRow(endRow),
		_startCol(startCol),
		_endCol(endCol)
	{
	}

	Matrix(const std::vector<std::vector<double>>& data) :
		_data(NDData::New(data)),
		_startRow(0),
		_endRow((int)data.size()),
		_startCol(0),
		_endCol((int)data[0].size())
	{
		// Must be square!
		for(int i=1;i<_endRow;++i)
			if(data[i].size()!=_endCol)
				throw RaggedVector();
	}
#pragma endregion

#pragma region Operators.

	// Greater than scalar.
	//
	Matrix operator > (const double v) const
	{
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,Rows(),0,Cols(),_data->Greater(v));
#else
		Matrix r(Rows(),Cols());
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				if(Value(i,j)>v)
					r.Value(i,j) = 1;
		return r;
#endif
	}

	// Divide by matrix.
	//
	Matrix operator / (const Matrix& v) const
	{
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0||v._startRow!=0||v._startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,Rows(),0,Cols(),_data->Div(v._data));
#else
		Matrix r(*this);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) /= v.Value(i,0);
		return r;
#endif
	}

	// Multiply by scalar.
	//
	Matrix operator * (const double v) const
	{
#ifdef PUSHDOWN
		return Matrix(0,Rows(),0,Cols(),_data->Mul((FP)v));
#else
		Matrix r(*this);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) *= v;
		return r;
#endif
	}

	// Multiply by matrix.
	//
	Matrix operator * (const Matrix& v) const
	{
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0||v._startRow!=0||v._startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,Rows(),0,Cols(),_data->Mul(v._data));
#else
		if(v.Rows()==1&&v.Cols()==1)
			return operator * (v.Value(0,0));
		if(Rows()==1&&Cols()==1)
			return v*Value(0,0);

		if(v.Rows()!=Rows()||v.Cols()!=Cols())
			throw IncompatibleShape();

		Matrix r(*this);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) *= v.Value(i,j);
		return r;
#endif	
	}

	// Inplace multiply by scalar.
	//
	void operator *= (double v)
	{
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				Value(i,j) *= (FP)v;
	}

	// Subtract scalar.
	//
	Matrix operator - (const double v) const
	{
		Matrix r(*this);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) -= (FP)v;
		return r;
	}

	// Subtract matrix.
	//
	Matrix operator - (const Matrix& v) const
	{
		if(v.Rows()!=Rows()||v.Cols()!=Cols())
			throw IncompatibleShape();

		Matrix r(*this);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) -= v.Value(i,j);
		return r;
	}

	// Inplace subtract matrix.
	//
	void operator -= (const Matrix& v)
	{
#ifdef PUSHDOWN
		return _data->_Sub(v._data);
#else
		if(r.Rows()!=Rows()||r.Cols()!=Cols())
			throw IncompatibleShape();

		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				Value(i,j) -= r.Value(i,j);
#endif
	}

	// Add scalar.
	//
	Matrix operator + (const double v) const
	{
		return Matrix(0,Rows(),0,Cols(),_data->Add((FP)v));
	}

	// Add matrix.
	//
	Matrix operator + (const Matrix& v) const
	{
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0||v._startRow!=0||v._startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,Rows(),0,Cols(),_data->Add(v._data));
#else
		if(v.Rows()!=Rows()||v.Cols()!=Cols())
			throw IncompatibleShape();

		Matrix r(*this);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) += v.Value(i,j);
		return r;
#endif	
	}

	// Unary minus.
	//
	Matrix operator - () const
	{
		Matrix r(Rows(),Cols());
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) = -Value(i,j);
		return r;
	}

	Matrix operator [] (const Matrix& v) const
	{
		Matrix r(v.Rows(),Cols());
		for(int j=0;j<Cols();++j)
			for(int i=0;i<v.Rows();++i)
				r.Value(i,j) = Value((int)v.Value(i,0),j);
		return r;
	}

	Matrix operator [] (const int& i)
	{
		return Matrix(_startRow+i,_startRow+i+1,_startCol,_endCol,_data);
	}

	void operator += (const Matrix& v)
	{
		if(Rows()!=v.Rows()||Cols()!=v.Cols())
			throw IncompatibleShape();
		for(int j=0;j<Cols();++j)
			for(int i=0;i<v.Rows();++i)
				Value(i,j) += v.Value(i,j);
	}

	void operator = (const Matrix& v)
	{
		_data = v._data;
		_startRow = v._startRow;
		_endRow = v._endRow;
		_startCol = v._startCol;
		_endCol = v._endCol;
	}

	bool operator != (const Matrix& r) const
	{
		if(_startRow!=r._startRow)
			return true;
		if(_endRow!=r._endRow)
			return true;
		if(_startCol!=r._startCol)
			return true;
		if(_endCol!=r._endCol)
			return true;
		if(*_data!=(*r._data))
			return true;
		return false;
	}

#pragma endregion

#pragma region Factories.

	// Identity matrix factory.
	//
	static Matrix Eye(int size)
	{
		Matrix r(size,size);
		for(int i=0;i<size;++i)
			r.Value(i,i) = 1.0;
		return r;
	}

	// Randomly initialised matrix factory.
	//
	static Matrix Rand(int rows,int cols)
	{
#ifdef PUSHDOWN
		NDArray data(new NDData({rows,cols}));
		data->Randomise();
		return Matrix(0,rows,0,cols,data);
#else
		Matrix r(rows,cols);
		for(int j=0;j<cols;++j)
			for(int i=0;i<rows;++i)
				r.Value(i,j) = rnd.NextDouble();
		return r;
#endif
	}

#pragma endregion

	int Rows() const
	{
		return _endRow-_startRow;
	}

	int Cols() const
	{
		return _endCol-_startCol;
	}
	
	FP& Value(int i,int j)
	{
		return _data->operator[]({_startRow+i,_startCol+j});
	}

	FP Value(int i,int j) const
	{
		return _data->operator[]({_startRow+i,_startCol+j});
	}

#pragma region Operations.

	// Column ArgMax returning scalar.
	//
	int ArgMax() const
	{
		if(Rows()!=1)
			throw IncompatibleShape();

		int mj = 0;
		for(int j=1;j<Cols();++j)
			if(Value(0,j)>Value(0,mj))
				mj = j;
		return mj;
	}

	// ArgMax along axis.
	//
	Matrix ArgMax(int axis) const
	{
		return Matrix(0,Rows(),0,1,_data->ArgMax(axis));
	}

	// Clip (scale) componented of the matrix by the norm.
	//
	void ClipNorm(double clipNorm)
	{
		_data->ClipNorm(clipNorm);
	}

	// Dot product/matrix multiplication.
	//
	Matrix Dot(const Matrix& v) const	
	{		
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0||v._startRow!=0||v._startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,Rows(),0,v.Cols(),_data->Dot(v._data));
#else
		if(v._data->Shape().size()!=_data->Shape().size())
			throw IncompatibleShape();
		if(Cols()!=v.Rows())
			throw IncompatibleShape();

		Matrix r(Rows(),v.Cols());
		for(int k=0;k<v.Cols();++k)
			for(int i=0;i<Rows();++i)
				for(int j=0;j<Cols();++j)
					r.Value(i,k) += Value(i,j)*v.Value(j,k);

		return r;
#endif
	}

	// Gather elements along axis.
	//
	Matrix Gather(int axis,const Matrix& indicies) const
	{
		if(axis==1)
		{
			if(indicies.Rows()!=Rows()||indicies.Cols()!=1)
				throw IncompatibleShape();
			Matrix r(Rows(),1);
			for(int i=0;i<Rows();++i)
				r.Value(i,0) = Value(i,(int)indicies.Value(i,0));
			return r;
		}
		throw "Not implemented!";
	}

	// Max value on axis.
	//
	Matrix Max(int axis) const
	{
		if(axis==1)
		{
			Matrix r(Rows(),1);
			for(int i=0;i<Rows();++i)
			{
				FP m = Value(i,0);
				for(int j=1;j<Cols();++j)
				{
					if(Value(i,j)>m)
						m = Value(i,j);
				}
				r.Value(i,0) = m;
			}
			return r;
		}
		throw "Not implemented!";
	}

	// Exponential.
	//
	Matrix Exp() const
	{
		Matrix r(*this);
		for(int j=0;j<r.Cols();++j)
			for(int i=0;i<r.Rows();++i)
				r.Value(i,j) = exp(r.Value(i,j));
		return r;
	}

	// Flatten to n rows and 1 column.
	//
	Matrix Flatten() const
	{
		Matrix r(Rows()*Cols(),1);
		int k = 0;
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(k++,0) = Value(i,j);
		return r;
	}

	// Elementwise log.
	//
	Matrix Log() const
	{
		Matrix r(Rows(),Cols());
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) = log(Value(i,j));
		return r;
	}

	// Global mean value.
	//
	Matrix Mean() const
	{
		FP sum = 0.0;
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				sum += Value(i,j);
		return Matrix(1,1,sum/(Rows()*Cols()));
	}

	// Mean along axis.
	//
	Matrix Mean(int axis) const
	{
		if(axis==1)
		{
			// Mean over columns (mean of columns for each row).
			Matrix r(Rows(),1);
			for(int i=0;i<Rows();++i)
			{
				FP sum = 0.0;
				for(int j=0;j<Cols();++j)
					sum += Value(i,j);
				r.Value(i,0) = sum/Cols();
			}
			return r;
		}
		else
			throw InvalidAxis();
	}

	// New matrix, same shape, initialised with 1.
	//
	Matrix Ones() const
	{
		return Matrix(Rows(),Cols(),1.0);
	}

	// Raise each element to the power n.
	//
	Matrix Pow(double exponent) const
	{
		if(_startRow!=0||_startCol!=0)
		{
			Matrix r(Rows(),Cols());
			for(int j=0;j<Cols();++j)
				for(int i=0;i<Rows();++i)
					r.Value(i,j) = pow(Value(i,j),exponent);
			return r;
		}
		else
			return Matrix(0,Rows(),0,Cols(),_data->Pow(exponent));
	}

	// Repeat n copies on axis.
	//
	Matrix Repeat(int copies,int axis) const
	{
		if(axis==0)
		{
			// Repeat each row n times.
			Matrix r(Rows()*copies,Cols());
			for(int i=0;i<Rows();++i)
			{
				for(int k=0;k<copies;++k)
				{
					for(int j=0;j<Cols();++j)
						r.Value((i*copies)+k,j) = Value(i,j);
				}
			}
			return r;
		}
		else if(axis==1)
		{
			// Repeat each column n times.
			Matrix r(Rows(),Cols()*copies);
			for(int j=0;j<Cols();++j)
			{
				for(int k=0;k<copies;++k)
				{
					for(int i=0;i<Rows();++i)
						r.Value(i,(j*copies)+k) = Value(i,j);
				}
			}
			return r;
		}
		else
			throw InvalidAxis();
	}

	// Reshape (without moving elements).
	//
	Matrix Reshape(int rows,int cols) const
	{
		if(cols==-1)
			cols = _data->Size()/rows;
		if(rows*cols!=_data->Size())
			throw IncompatibleShape();		
		
		Matrix r(rows,cols);
		r._data = std::make_shared<NDData>(*_data);
		return r;
	}

	// Elementwise square root.
	//
	Matrix Sqrt() const
	{
		return Matrix(0,Rows(),0,Cols(),_data->Sqrt());
	}

	// Global sum.
	//
	Matrix Sum() const
	{
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,1,0,1,_data->Sum());
#else
		Matrix r(1,1);
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(0,0) += Value(i,j);
		return r;
#endif
	}

	// Sum along axis.
	//
	Matrix Sum(int axis,bool keepdims=false) const
	{
/*
* // Tricky becuase of different sized matrix returned.
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0)
			throw IncompatibleShape();
		return Matrix()
#else
*/
		if(axis<0)
			axis = ((int)_data->Shape().size())+axis;
		if(axis==0)
		{
			// Sum rows to 1 row.
			Matrix r(1,Cols());
			for(int j=0;j<Cols();++j)
				for(int i=0;i<Rows();++i)
					r.Value(0,j) += Value(i,j);
			
			// If 'keepdims' is false then theoretically the axis along which the sum was performed is removed because it's only got 1 entry.
			// For axis=rows this means the row axis is removed making the column axis the row axis. Since this Matrix class has to be 2D
			// transposing the matrix probably has a similar effect.
			if(keepdims==false)
				r = r.Transpose();
			return r;
		}
		else if(axis==1)
		{
			// Sum columns to 1 column.
			Matrix r(Rows(),1,0.0);
			for(int j=0;j<Cols();++j)
				for(int i=0;i<Rows();++i)
					r.Value(i,0) += Value(i,j);
			return r;
		}
		else
			throw InvalidAxis();
//#endif
	}

	// Elementwise tanh.
	//
	Matrix Tanh() const
	{
		Matrix r(Rows(),Cols());
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(i,j) = tanh(Value(i,j));
		return r;
	}

	// Transpose (rows become columns, columns become rows, elements moved).
	//
	Matrix Transpose() const
	{
#ifdef PUSHDOWN
		if(_startRow!=0||_startCol!=0)
			throw IncompatibleShape();
		return Matrix(0,Cols(),0,Rows(),_data->Transpose());
#else
		Matrix r(Cols(),Rows());
		for(int j=0;j<Cols();++j)
			for(int i=0;i<Rows();++i)
				r.Value(j,i) = Value(i,j);
		return r;
#endif
	}

	// New matrix, same shape, initialised with 0.
	//
	Matrix Zeros() const
	{
		return Matrix(Rows(),Cols(),0.0);
	}

#pragma endregion

	// Load from file.
	//
	static Matrix Load(const std::string& filename)
	{
		Matrix r(1,1,0);
		std::fstream file;
		file.open(filename,std::ios::in);
		if(file.is_open())
		{
			file>>std::noskipws;
			char c;
			file>>c;
			if(c=='[')
			{
				// Row or column array.
				file>>c;
				std::vector<std::vector<double>> data = Read1Or2DArray(file,c);
				r = Matrix(data);
			}
			else
				throw UnexpectedCharacter();
			file.close();
		}
		else
			throw FileNotFound();
		return r;
	}

	// Print to console (stdout).
	//
	void Print() const
	{
		for(int i=0;i<Rows();++i)
		{
			if(i==0)
				std::cout<<"[";
			for(int j=0;j<Cols();++j)
			{
				if(j==0)
					std::cout<<"[";
				else
					std::cout<<",";
				std::cout<<Value(i,j);
				if(j==Cols()-1)
					std::cout<<"]";
			}
			if(i==Rows()-1)
				std::cout<<"]";
			else
				std::cout<<",";
			std::cout<<std::endl;
		}
	}

	const std::vector<int>& Shape() const
	{
		return _data->Shape();
	}
};
