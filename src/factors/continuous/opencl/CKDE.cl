#line 1 "src/factors/continuous/opencl/CKDE.cl.src"

/*
 *****************************************************************************
 **       This file was autogenerated from a template  DO NOT EDIT!!!!      **
 **       Changes should be made to the original source (.src) file         **
 *****************************************************************************
 */

#line 1
/* This code assumes column major data for matrices. */

#define IDX(i, j, rows) (i) + ((j)*(rows))
#define ROW(idx, rows) (idx) % (rows)
#define COL(idx, rows) (idx) / (rows)

#line 10

__kernel void substract_matrix_vec_double(__constant double *matrix,
                                   __private uint matrix_rows,
                                   __private uint matrix_cols,
                                   __constant double *vec_location,
                                   __private uint vec_location_rows,
                                   __private uint vec_row_idx,
                                   __global double *res
                                )
{
    int i = get_global_id(0);

    int r = ROW(i, matrix_rows);
    int c = COL(i, matrix_rows);

    res[IDX(r, c, matrix_rows)] = matrix[i] - vec_location[IDX(vec_row_idx, c, vec_location_rows)];
}


__kernel void solve_double(__global double *diff_matrix, 
                    __private uint diff_matrix_rows, 
                    __private uint diff_matrix_cols,
                    __constant double *cholesky_matrix, 
                    __private uint cholesky_dim) {
    uint r = get_global_id(0);
    
    for (uint c = 0; c < diff_matrix_cols; c++) {
        for (uint i = 0; i < c; i++) {
            diff_matrix[IDX(r, c, diff_matrix_rows)] -= cholesky_matrix[IDX(c, i, cholesky_dim)] * diff_matrix[IDX(r, i, diff_matrix_rows)];
        }
        diff_matrix[IDX(r, c, diff_matrix_rows)] /= cholesky_matrix[IDX(c, c, cholesky_dim)];
    }
}

__kernel void square_double(__global double *m) {
    uint idx = get_global_id(0);
    double d = m[idx];
    m[idx] = d * d;
}


#line 10

__kernel void substract_matrix_vec_float(__constant float *matrix,
                                   __private uint matrix_rows,
                                   __private uint matrix_cols,
                                   __constant float *vec_location,
                                   __private uint vec_location_rows,
                                   __private uint vec_row_idx,
                                   __global float *res
                                )
{
    int i = get_global_id(0);

    int r = ROW(i, matrix_rows);
    int c = COL(i, matrix_rows);

    res[IDX(r, c, matrix_rows)] = matrix[i] - vec_location[IDX(vec_row_idx, c, vec_location_rows)];
}


__kernel void solve_float(__global float *diff_matrix, 
                    __private uint diff_matrix_rows, 
                    __private uint diff_matrix_cols,
                    __constant float *cholesky_matrix, 
                    __private uint cholesky_dim) {
    uint r = get_global_id(0);
    
    for (uint c = 0; c < diff_matrix_cols; c++) {
        for (uint i = 0; i < c; i++) {
            diff_matrix[IDX(r, c, diff_matrix_rows)] -= cholesky_matrix[IDX(c, i, cholesky_dim)] * diff_matrix[IDX(r, i, diff_matrix_rows)];
        }
        diff_matrix[IDX(r, c, diff_matrix_rows)] /= cholesky_matrix[IDX(c, c, cholesky_dim)];
    }
}

__kernel void square_float(__global float *m) {
    uint idx = get_global_id(0);
    double d = m[idx];
    m[idx] = d * d;
}



