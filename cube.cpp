#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simplification and\n"
    "memory storage to find visible faces of a cube in a list of images\n"
    "Returns sequence of quadrilaterals detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./cube_detection\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}

int thresh = 50, N = 5;
const char* wndname = "Cube Detection Demo";

// returns sequence of quadrilaterals detected on the image
static void findQuadrilaterals(const Mat& image, vector<vector<Point>>& quadrilaterals)
{
    quadrilaterals.clear();

    Mat timg(image);
    medianBlur(image, timg, 9);
    Mat gray0(timg.size(), CV_8U), gray;

    vector<vector<Point>> contours;

    // find quadrilaterals in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for (int l = 0; l < N; l++)
        {
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 5, thresh, 5);
                // dilate canny output to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
            }
            else
            {
                // apply threshold if l != 0
                gray = gray0 >= (l + 1) * 255 / N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++)
            {
                // approximate contour with accuracy proportional to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

                // quadrilateral contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                if (approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)))
                {
                    quadrilaterals.push_back(approx);
                }
            }
        }
    }
}

// the function draws all the quadrilaterals in the image
static void drawQuadrilaterals(Mat& image, const vector<vector<Point>>& quadrilaterals)
{
    for (size_t i = 0; i < quadrilaterals.size(); i++)
    {
        const Point* p = &quadrilaterals[i][0];

        int n = (int)quadrilaterals[i].size();
        if (p->x > 3 && p->y > 3)
            polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
    }
}

// checks if two quadrilaterals share an edge
bool sharesEdge(const vector<Point>& quad1, const vector<Point>& quad2)
{
    int sharedPoints = 0;
    for (const Point& pt1 : quad1)
    {
        for (const Point& pt2 : quad2)
        {
            if (norm(pt1 - pt2) < 1e-6) // if points are the same
            {
                sharedPoints++;
            }
        }
    }
    return sharedPoints == 2; // two quads share an edge if they have exactly 2 common points
}

// detect if a cube is formed by the detected quadrilaterals
bool detectCube(const vector<vector<Point>>& quadrilaterals)
{
    for (size_t i = 0; i < quadrilaterals.size(); i++)
    {
        for (size_t j = i + 1; j < quadrilaterals.size(); j++)
        {
            if (sharesEdge(quadrilaterals[i], quadrilaterals[j]))
            {
                for (size_t k = j + 1; k < quadrilaterals.size(); k++)
                {
                    if (sharesEdge(quadrilaterals[i], quadrilaterals[k]) ||
                        sharesEdge(quadrilaterals[j], quadrilaterals[k]))
                    {
                        return true; // found a cube
                    }
                }
            }
        }
    }
    return false;
}

int main(int /*argc*/, char** /*argv*/)
{
    static const char* names[] = {"positive_cube.png", "imgs/manyStickies.jpg", 0};
    help();
    namedWindow(wndname, 1);
    vector<vector<Point>> quadrilaterals;

    for (int i = 0; names[i] != 0; i++)
    {
        Mat image = imread(names[i], 1);
        if (image.empty())
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findQuadrilaterals(image, quadrilaterals);
        drawQuadrilaterals(image, quadrilaterals);

        if (detectCube(quadrilaterals))
        {
            cout << "Cube detected in image: " << names[i] << endl;
            putText(image, "Cube Detected", Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        imshow(wndname, image);
        int c = waitKey();
        if ((char)c == 27)
            break;
    }

    return 0;
}

