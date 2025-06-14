#pragma once

#include <random>
#include "DType.h"
#include "Exceptions.h"


class UniformDistribution
{
	std::random_device					_rd;
    std::mt19937						_gen;
	std::uniform_int_distribution<>		_rndInt;
	std::uniform_real_distribution<FP>	_rndReal;

public:
	UniformDistribution() :
		_gen(0/*_rd()*/),
		_rndReal(0,1)
	{		
	}

	// Returns a random number >=0 and <max, uniformly distributes [0,max).
	//
	int NextInt(int max)	// 0 through max exclusive.
	{
		const FP rnd = Sample();
		const int rndInt = (int)(rnd*max);
		return rndInt;
	}

	// Returns a random number >=0 and <1, uniformly distributed [0,1).
	//
	FP Sample()
	{
		return _rndReal(_gen);
	}

	int NextInt()
	{
		return _rndInt(_gen);
	}

	std::mt19937& Generator()
	{
		return _gen;
	}
};


class NormalDistribution
{
	std::random_device					_rd;
    std::mt19937						_gen;
	std::normal_distribution<double>	_rnd;

public:
	NormalDistribution(double mu,double sigma) :
		_gen(_rd()),
		_rnd(mu,sigma)
	{		
	}

	// Samples from the normal distribution.
	//
	FP Sample()
	{
		return _rnd(_gen);
	}
};


extern UniformDistribution rnd;

