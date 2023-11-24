// main.c
#include <stdio.h>
#include <opencv2/opencv.hpp>

typedef unsigned char uchar;

extern void cudaFunction(uchar *hnpixels, uchar *hpixels, int height, int width);

void save(uchar *pixels, int height, int width, const char *name, const char *type){
    cv::Mat image(height, width, CV_8UC3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b pixel;
            for (int k = 0; k < 3; k++) {
                pixel[k] = pixels[i*width + j + height*width * k];
            }
            image.at<cv::Vec3b>(i, j) = pixel;
        }
    }
    char *path = (char *)malloc((strlen(name) + strlen(type)) * sizeof(char));
    strcpy(path, type);
    strcat(path, name);
    cv::imwrite(path, image);
    free(path);
}

int main(int argc, char** argv) {

    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (image.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    int height = image.rows;
    int width = image.cols;
    int size = height * width;

    uchar *pixels = (uchar*)malloc(size * 3 * sizeof(uchar));
    uchar *npixels = (uchar*)malloc(2*2 * size * 3 * sizeof(uchar));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; k++) {
                pixels[i*width + j + size * k] = pixel[k];
            }
        }   
    }
    
    cudaFunction(npixels, pixels, height, width);

    save(npixels, height * 2, width * 2, argv[1], "gpu_bicubic_");
    free(pixels);
    free(npixels);
    return 0;
}