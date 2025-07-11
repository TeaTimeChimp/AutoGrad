#pragma once


constexpr int tile_size = 32;	// Tile size for matrix multiplication.


// Adds the whole square tile to an arbitray shape matrix.
//	Both source tile and destination matrix are row major (column stride is 1).
//	Both source and destination matrices are 64-byte aligned.
static void add_from_whole_tile(
	const float* const tile,
	float* const dst,
	const int dst_row_stride,const int dst_i,const int dst_j)
{
	float* const dst_begin = dst+(dst_row_stride*dst_i)+dst_j;
	const float* tile_row = tile;
	for(int i=0;i<tile_size;++i)
	{
		float* const dst_row = dst_begin+i*dst_row_stride;
		for(int j=0;j<tile_size;j+=8)
		{
			__m256 v_tile = _mm256_load_ps(&tile_row[j]);   // aligned load from tile
			__m256 v_dst  = _mm256_load_ps(&dst_row[j]);    // aligned load from dst
			__m256 v_sum  = _mm256_add_ps(v_dst,v_tile);	// element-wise add
			_mm256_store_ps(&dst_row[j],v_sum);             // aligned store to dst
		}
		tile_row += tile_size;
	}
}


// Adds a partial area from a square tile to an arbitray shape matrix.
//
static void add_from_partial_tile(
	const float* const tile,
	float* const dst, const int dst_row_stride,
	const int dst_i, const int dst_j,
	const int rows, const int cols)
{
	float* const dst_begin = dst+(dst_row_stride*dst_i)+dst_j;
	float* dst_row_begin = dst_begin;
	for(int i=0;i<rows;++i)
	{
		float* dst_value = dst_row_begin;
		for(int j=0;j<cols;++j)
			dst_value[j] += tile[i*tile_size+j];
		dst_row_begin += dst_row_stride;
	}
}


// Adds a square tile to an arbitray shape matrix.
//	Tile and destination must be row major (column stride is 1).
//
static void add_from_tile(
	const float* const tile,
	float* const dst, const int dst_rows, const int dst_cols, const int dst_row_stride,
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
			dst, dst_row_stride,
			dst_i,dst_j);
	else
		add_from_partial_tile(
			tile,
			dst, dst_row_stride,
			dst_i,dst_j,
			rows,cols);
}


// Copies square tile from and arbitrary shape matrix.
//	Source and tile are row major (column stride is 1).
//
static void copy_whole_tile_row_major(
	float* const tile,
	const float* const src,const int src_row_stride,const int src_i,const int src_j)
{
	const float* const src_begin = src+(src_row_stride*src_i)+src_j;	
	const float* src_row_begin = src_begin;
	for(int i=0;i<tile_size;++i)
	{
        // Use AVX2 to copy 8 floats at a time
		float* const dst_value = &tile[i*tile_size];
        for(int j=0;j<tile_size;j+=8) 
		{
			__m256 v = _mm256_loadu_ps(&src_row_begin[j]);
			_mm256_store_ps(&dst_value[j],v);
        }
		src_row_begin += src_row_stride;
	}
}


// Copies a partial tile area from an arbitrary shape matrix.
//   Source and destination matrices are row major (column stride is 1).
//
static void copy_partial_tile(
	float* const tile,
	const float* const src,const int src_row_stride,const int src_i,const int src_j,
	const int rows, const int cols)
{
	memset(tile,0,sizeof(float)*tile_size*tile_size);
	const float* const src_begin = src+(src_row_stride*src_i)+src_j;
	const float* src_row_begin = src_begin;
	for(int i=0;i<rows;++i)
	{
		const float* src_value = src_row_begin;
		for(int j=0;j<cols;++j)
			tile[i*tile_size+j] = src_row_begin[j];
		src_row_begin += src_row_stride;
	}
}


// Copies a square tile from an arbitrary shape matrix.
//   Source and destination matrices are row major (column stride is 1).
//
static void copy_to_tile(
	float* const tile,
	const float* const src, const int src_rows, const int src_cols, const int src_row_stride,
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
		copy_whole_tile_row_major(
			tile,
			src,src_row_stride,src_i,src_j);
	else
		copy_partial_tile(
			tile,
			src,src_row_stride,src_i,src_j,
			rows,cols);
}


// Multiply 2 square matricies a and bT to produce c.
//   'a' is row major (values across a row are adjacent).
//   'bT' is column major (values down a column are adjacent).
//   All pointers are 64-byte aligned.
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
