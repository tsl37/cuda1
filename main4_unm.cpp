// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
void cu_run_grayscale( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img );
void cu_run_filter( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img , uchar3 mask);
void cu_run_split( CudaImg t_color_cuda_img, CudaImg t_r_cuda_img, CudaImg t_g_cuda_img, CudaImg t_b_cuda_img);

int main( int t_numarg, char **t_arg )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    if ( t_numarg < 2 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Load image
    cv::Mat l_bgr_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR ); // CV_LOAD_IMAGE_COLOR );

    if ( !l_bgr_cv_img.data )
    {
        printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
        return 1;
    }

    // create empty BW image
    cv::Mat l_bw_cv_img( l_bgr_cv_img.size(), CV_8UC3 );

    cv::Mat l_b_cv_img( l_bgr_cv_img.size(), CV_8UC3 );
    cv::Mat l_g_cv_img( l_bgr_cv_img.size(), CV_8UC3 );
    cv::Mat l_r_cv_img( l_bgr_cv_img.size(), CV_8UC3 );

    // data for CUDA
    CudaImg l_bgr_cuda_img, l_bw_cuda_img, l_r_cuda_img,l_g_cuda_img,l_b_cuda_img;
    l_bgr_cuda_img.m_size.x = l_bw_cuda_img.m_size.x = l_bgr_cv_img.size().width;
    l_bgr_cuda_img.m_size.y = l_bw_cuda_img.m_size.y = l_bgr_cv_img.size().height;
    l_bgr_cuda_img.m_p_uchar3 = ( uchar3 * ) l_bgr_cv_img.data;
    l_bw_cuda_img.m_p_uchar3 = ( uchar3 * ) l_bw_cv_img.data;

    l_r_cuda_img.m_size.x =
    l_g_cuda_img.m_size.x =
    l_b_cuda_img.m_size.x =l_bgr_cv_img.size().width;

    l_r_cuda_img.m_size.y =
    l_g_cuda_img.m_size.y =
    l_b_cuda_img.m_size.y =l_bgr_cv_img.size().height;
    
    l_r_cuda_img.m_p_uchar3 = (uchar3 *) l_r_cv_img.data;
    l_g_cuda_img.m_p_uchar3 = (uchar3 *) l_g_cv_img.data;
    l_b_cuda_img.m_p_uchar3 = (uchar3 *) l_b_cv_img.data;
    uchar3 mask;
    mask.x = 1;
    mask.y = 0;
    mask.z = 0;
    // Function calling from .cu file
    //cu_run_filter( l_bgr_cuda_img, l_bw_cuda_img,mask );

    cu_run_split(l_bgr_cuda_img, l_r_cuda_img,  l_g_cuda_img, l_b_cuda_img);

    // Show the Color and BW image
    cv::imshow( "Color", l_bgr_cv_img );
    cv::imshow( "Red", l_r_cv_img );
    cv::imshow( "Green", l_g_cv_img );
    cv::imshow( "Blue", l_b_cv_img );
    cv::waitKey( 0 );
}

