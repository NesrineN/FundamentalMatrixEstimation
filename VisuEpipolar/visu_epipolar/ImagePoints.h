#ifndef IMAGEPOINTS_H
#define IMAGEPOINTS_H

#include "ImageZoom.h"
#include "libNumerics/matrix.h"
#include <vector>

struct Point {
    float x, y;
    QColor col;
    Point(float x0, float y0, QColor c): x(x0), y(y0), col(c) {}
};

class ImagePoints : public ImageZoom {
    Q_OBJECT
public:
    ImagePoints(QWidget* parent=0);

    std::vector<Point> pts;

    void draw_line(const libNumerics::vector<float>& l);
    void set_selection(int i);
protected:
    int select;
    void draw_point(QPainter& paint, const Point& pt);
    virtual void pixmap_from_image();
    virtual void update_status(int x, int y);

signals:
    void new_selection(int select); // New point selection
    void new_pos(float x, float y); // New mouse position
protected: // Events handlers
    virtual void mousePressEvent(QMouseEvent*);
    virtual void mouseMoveEvent(QMouseEvent*);
    virtual void mouseReleaseEvent(QMouseEvent*);
};

#endif
