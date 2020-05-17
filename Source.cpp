/*
* Auteurs:
    - Marco Castro
    - Berrahal Kaouter
* Date: 2020-04-21
* Information: Analyse d’un escalier : comptage et détection des marches
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

//La structure trie 2 points
struct myclass {
    bool operator() (cv::Point pt1, cv::Point pt2) {
        return (pt1.y < pt2.y);
    }
} myobject;

//Unification de points très proches dans les coordonnées X et Y
std::vector<Point> unification(std::vector<Point> p, int x, int y) {
    for (int i = 0; i < p.size(); i++)
        for (int j = i; j < p.size(); j++)
            if (abs(p[i].x - p[j].x) < x || abs(p[i].y - p[j].y) < y)
                p.erase(p.begin() + j);
    return p;
}

int main() {

    //Gradient
    Mat src, src_gray, grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    src = imread("img/im1.Jpg", 1);

    if (!src.data) {
        std::cout << "Impossible d'ouvrir ou de trouver l'image" << "\n";
        return -1;
    }

    if (src.cols > 2000 || src.rows > 2000) {
        cv::resize(src, src, cv::Size(0, 0), 0.15, 0.15);
    }
    else if (src.cols > 650 || src.rows > 650) {

        cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5);
    }

    imshow("Escalier", src);

    moveWindow("Escalier", 0, 0);

    GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Convert it to gray

    cvtColor(src, src_gray, COLOR_RGBA2GRAY, 0);

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    // Gradient X
    Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    // Gradient Y
    Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    // Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);// si incremente un ou deux la taille de la ligne etc 'est po bon on aura bcq de ligne
// po de dilatation pour ne po avoir d'augmenation 
    imshow("Sobel", grad);

    //Standard Hough Line Transform
    Mat dst, color_dst;

    cv::threshold(grad, color_dst, 0, 255, THRESH_BINARY | THRESH_OTSU);


    std::vector<Vec4i> linesP; // will hold the results of the detection
    
    HoughLinesP(color_dst, linesP, 1, CV_PI / 180, 50, 70, 10); // runs the actual detection

    imshow("Line Transform", color_dst);

    // Draw the lines
    std::vector<Point> p1;
    Mat points = cv::Mat::zeros(color_dst.rows, color_dst.cols, CV_8UC1);
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        //On trace tous les segments parallèles inférieurs à 40 degrése           
        if (abs(atan2(l[3] - l[1], l[2] - l[0])) < 0.15) {
            line(points, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1, 8, 0);
            //Le vecteur de points p1 stocke les points des lignes parallèles
            p1.push_back(Point(l[0], l[1]));
        }
    }

    //Les points du vecteur p1 sont classés par ordre croissant
    std::sort(p1.begin(), p1.end(), myobject);

    //On utilise la fonction d'unification 3 fois. L'objectif est de trouver le point le plus extrême, 
    //c'est-à-dire le plus proche de la coordonnée X et le plus éloigné de la coordonnée Y

    //Les nouveaux points représentent le nombre de coin des marches. 
    //Le vecteur de Point p2 stock ces coordonnées

    std::vector<Point> p2;
    p2 = unification(p1, 13, 13);

    std::cout << "Nombre de marches: " << p2.size() << "\n";

    for (int j = 0; j < p2.size(); j++)
        cv::circle(points, p2[j], 4, cv::Scalar(200, 0, 0), -1);


    imshow("Marches", points);

    //2ème unification de points. L'objectif est de nettoyer les lignes parallèles de chaque étape
    //Le vecteur de Point p3 stock ces coordonnées

    std::vector<Point> p3;    Mat points2 = cv::Mat::zeros(color_dst.rows, color_dst.cols, CV_8UC1);
    p3 = unification(unification(unification(p1, 0, 10), 0, 10), 0, 11);


    

    //std::cout << "Points apres unification " << "\n";
    //On calcule la direction de l'escalier
    //La méthode consiste à calculer des points entre les marches de l'axe Y.
    //S'il y a un ou plusieurs points entre 2 marches (du vecteur p2), c'est un escalier vers le haut. 
    //S'il n'y a aucun point, c'est un escalier vers le bas.
    
    int direction = 0;
    for (int i = 0; i < p3.size(); i++) {
        //std::cout << "Point " << i << " :" << p3[i] << " " << "\n";
        cv::circle(points2, p3[i], 4, cv::Scalar(200, 0, 0), -1);
        line(points2, Point(p3[i].x, p3[i].y), Point(p3[i].x + 100, p3[i].y), Scalar(255, 0, 0), 1, 8, 0);
        if (p2[p2.size() - 1].y > p3[i].y&& p3[i].y > p2[p2.size() - 2].y)
            direction++;
    }

    if (direction == 0)
        std::cout << "\nEscalier vers le bas " << "\n";
    else if (direction > 0)
        std::cout << "\nEscalier vers le haut " << "\n";

    //imshow("Direction", points2);

    cv::waitKey(0);
    return 0;
}