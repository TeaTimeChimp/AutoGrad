#pragma once


constexpr int tile_size = 32;	// Tile size for matrix multiplication.


// Adds the whole square tile to an arbitray shape matrix.
//
static void add_from_whole_tile(
	const float* const tile,
	float* const dst, const int dst_row_stride, const int dst_col_stride,
	const int dst_i, const int dst_j)
{
	float* const dst_begin = dst+(dst_row_stride*dst_i)+(dst_col_stride*dst_j);
	float* dst_row_begin = dst_begin;
	for(int i=0;i<tile_size;++i)
	{
		float* dst_value = dst_row_begin;
		for(int j=0;j<tile_size;++j)
		{
			*dst_value += tile[i*tile_size+j];
			dst_value += dst_col_stride;
		}
		dst_row_begin += dst_row_stride;
	}
}


// Adds a partial area from a square tile to an arbitray shape matrix.
//
static void add_from_partial_tile(
	const float* const tile,
	float* const dst, const int dst_row_stride, const int dst_col_stride,
	const int dst_i, const int dst_j,
	const int rows, const int cols)
{
	float* const dst_begin = dst+(dst_row_stride*dst_i)+(dst_col_stride*dst_j);	
	float* dst_row_begin = dst_begin;
	for(int i=0;i<rows;++i)
	{
		float* dst_value = dst_row_begin;
		for(int j=0;j<cols;++j)
		{
			*dst_value += tile [i*tile_size+j];
			dst_value += dst_col_stride;
		}
		dst_row_begin += dst_row_stride;
	}
}


// Adds a square tile to an arbitray shape matrix.
//
static void add_from_tile(
	const float* const tile,
	float* const dst, const int dst_rows, const int dst_cols, const int dst_row_stride, const int dst_col_stride,
	const int dst_i, const int dst_j)
{
	// Assume whole tile fits in the source and destination matrices.
	int rows = tile_size;
	int cols = tile_size;

	// Clip the tile to the destination matrix.
	if(dst_i+tile_size>dst_rows)
		rows = dst_rows-dst_i;
	if(dst_j+tile_size>dst_cols)
		cols = dst_cols-dst_j;

	// Use the whole tile if it fits, otherwise use a partial tile.
	if(rows==tile_size&&cols==tile_size)
		add_from_whole_tile(
			tile,
			dst, dst_row_stride, dst_col_stride,
			dst_i,dst_j);
	else
		add_from_partial_tile(
			tile,
			dst, dst_row_stride, dst_col_stride,
			dst_i,dst_j,
			rows,cols);
}


// Copies square tile from and arbitrary shape matrix - source row major (values across a row are adjacent).
//
static void copy_whole_tile_row_major(
	float* const tile,
	const float* const src,const int src_row_stride, const int _src_col_stride,const int src_i,const int src_j)
{
	_ASSERT(_src_col_stride==1);
	constexpr int src_col_stride = 1;
	const float* const src_begin = src+(src_row_stride*src_i)+(src_col_stride*src_j);	
	const float* src_row_begin = src_begin;
	for(int i=0;i<tile_size;++i)
	{
        // Use AVX2 to copy 8 floats at a time
		const float* src_value = src_row_begin;
        for(int j=0;j<tile_size;j+=8) 
		{
            __m256 v = _mm256_set_ps(
                src_value[7*src_col_stride],
                src_value[6*src_col_stride],
                src_value[5*src_col_stride],
                src_value[4*src_col_stride],
                src_value[3*src_col_stride],
                src_value[2*src_col_stride],
                src_value[1*src_col_stride],
                src_value[0*src_col_stride]
            );
			_mm256_storeu_ps(&tile[i*tile_size+j],v);	// Tile is row major.
            src_value += 8*src_col_stride;
        }
		src_row_begin += src_row_stride;
	}
}


// Copies square tile from and arbitrary shape matrix - source column major (values down a column are adjacent).
//
static void copy_whole_tile_col_major(
	float* const tile,
	const float* const src,const int _src_row_stride, const int src_col_stride,const int src_i,const int src_j)
{
	_ASSERT(_src_row_stride==1);
	constexpr int src_row_stride = 1;
	const float* const src_begin = src+(src_row_stride*src_i)+(src_col_stride*src_j);	
	const float* src_col_begin = src_begin;
	for(int j=0;j<tile_size;++j)
	{
		for(int i=0;i<tile_size;++i)
			tile[i*tile_size+j] = src_col_begin[i*src_row_stride];	// Tile is row major.
		src_col_begin += src_col_stride;
	}
}


// Copies square tile from and arbitrary shape matrix.
//
static void copy_whole_tile(
	float* const tile,
	const float* const src,const int src_row_stride, const int src_col_stride,const int src_i,const int src_j)
{
	if(src_col_stride<src_row_stride)
		copy_whole_tile_row_major(
			tile,
			src,src_row_stride,src_col_stride,src_i,src_j);
	else
		copy_whole_tile_col_major(
			tile,
			src,src_row_stride,src_col_stride,src_i,src_j);
}


// Copies a partial tile area from an arbitrary shape matrix.
//
static void copy_partial_tile(
	float* const tile,
	const float* const src,const int src_row_stride,const int src_col_stride,const int src_i,const int src_j,
	const int rows, const int cols)
{
	memset(tile,0,sizeof(float)*tile_size*tile_size);
	const float* const src_begin = src+(src_row_stride*src_i)+(src_col_stride*src_j);
	const float* src_row_begin = src_begin;
	for(int i=0;i<rows;++i)
	{
		const float* src_value = src_row_begin;
		for(int j=0;j<cols;++j)
			tile[i*tile_size+j] = src_row_begin[j*src_col_stride];
		src_row_begin += src_row_stride;
	}
}


// Copies a square tile from an arbitrary shape matrix.
//
static void copy_to_tile(
	float* const tile,
	const float* const src, const int src_rows, const int src_cols, const int src_row_stride, const int src_col_stride,
	const int src_i, const int src_j)
{
	// Assume whole tile is to be copied from the source matrix.
	int rows = tile_size;
	int cols = tile_size;

	// Clip the copy to the source matrix.
	if(src_i+tile_size>src_rows)
		rows = src_rows-src_i;
	if(src_j+tile_size>src_cols)
		cols = src_cols-src_j;

	// Use the whole tile if it fits, otherwise use a partial tile.
	if(rows==tile_size&&cols==tile_size)
		copy_whole_tile(
			tile,
			src,src_row_stride,src_col_stride,src_i,src_j);
	else
		copy_partial_tile(
			tile,
			src, src_row_stride, src_col_stride,src_i,src_j,
			rows,cols);
}


// Multiply 2 square matricies a and bT to produce c.
//   'a' is row major (values across a row are adjacent).
//   'bT' is column major (values down a column are adjacent).
// 
// The compiler will unroll the inner loop and use AVX2 256 bit SIMD fused multiply add instructons.
//
static void matmul_tile(const float* const a,const float* const bT,float* const c)
{
	for(int i=0;i<tile_size;++i)
	{
		for(int j=0;j<tile_size;++j)
		{
			float acc = 0.0;
			for(int k=0;k<tile_size;++k)
				acc += a[i*tile_size+k] * bT[j*tile_size+k];
			c[i*tile_size+j] = acc;
		}
	}
}
