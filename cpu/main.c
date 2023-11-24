#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <string.h>

void bicubic(cv::Vec3b** bicubic_pixels, cv::Vec3b** pixels, int height, int width){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(i == 0 || i == height - 1 || j == 0 || j == width -1){
                bicubic_pixels[i][j] = pixels[i >> 1][j >> 1];
                continue;
            }
            unsigned int suma[] = {0, 0, 0};
            for(int k = -1; k < 2; k++){
                for(int l = -1; l < 2; l++){
                    for(int m = 0; m < 3; m++){
                        suma[m] += pixels[(i + k) >> 1][(j + l) >> 1 ][m];
                    }
                }
            }
            for(int m = 0; m < 3; m++){
                bicubic_pixels[i][j][m] = (unsigned char)(suma[m] / 9);
            }
        }
    }
}

void save(cv::Vec3b **pixels, int height, int width, const char *name, const char *type){
    cv::Mat image(height, width, CV_8UC3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b pixel = pixels[i][j];
            image.at<cv::Vec3b>(i, j) = pixel;
        }
    }
    char *path = (char *)malloc(strlen(name) + strlen(type));
    strcpy(path, type);
    strcat(path, name);
    cv::imwrite(path, image);
    free(path);
}

void bicubic_interpolation(cv::Mat *image, const char *path){
    clock_t start, end;
    int height = image->rows;
    int width = image->cols;

    cv::Vec3b** pixels = new cv::Vec3b*[height];
    cv::Vec3b** bicubic_pixels = new cv::Vec3b*[2 * height];
    for (int i = 0; i < height; i++) {
        pixels[i] = new cv::Vec3b[width];
        bicubic_pixels[i] = new cv::Vec3b[2 * width];
        bicubic_pixels[height + i] = new cv::Vec3b[2 * width];
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            pixels[i][j] = image->at<cv::Vec3b>(i, j);
        }
    }
    start = clock();
        bicubic(bicubic_pixels, pixels, 2 * height, 2 * width);
    end = clock();

    double time_taken = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken: %fs\n", time_taken);

    save(bicubic_pixels, 2 * height, 2 * width, path, "cpu_bicubic_");

    for (int i = 0; i < height; i++) {
        delete[] pixels[i];
        delete[] bicubic_pixels[i];
        delete[] bicubic_pixels[height + i];
    }
    delete[] pixels;
    delete[] bicubic_pixels;
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

    bicubic_interpolation(&image, argv[1]);

    return 0;
}