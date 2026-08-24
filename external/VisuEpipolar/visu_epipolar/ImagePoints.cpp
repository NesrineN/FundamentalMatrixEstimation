#include "ImagePoints.h"

#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtWidgets/QDoubleSpinBox>

#include <cmath>
#include <string>
#include <iostream>

const int SELECT_RADIUS=20; ///< Radius for point selection

// Constructor.
ImagePoints::ImagePoints(QWidget* parent)
: ImageZoom(parent), select(-1) {
}

void ImagePoints::pixmap_from_image() {
    ImageZoom::pixmap_from_image();
    QPainter p(&pixmap());
    for(size_t i=0; i<pts.size(); i++)
        draw_point(p, pts[i]);
}

void ImagePoints::draw_point(QPainter& p, const Point& pt) {
    p.setPen(pt.col);
    p.setBrush(pt.col);
    float f = static_cast<float>(scale());
    int x = static_cast<int>(f*(pt.x - rectImage().x())+.5f);
    int y = static_cast<int>(f*(pt.y - rectImage().y())+.5f);
    p.drawEllipse(x-3/2, y-3/2, 3, 3);
}

void ImagePoints::set_selection(int i) {
    select = i;
    update_view();
    update();
}

void ImagePoints::mousePressEvent(QMouseEvent* pEvent) {
    if(pEvent->button() != Qt::LeftButton ||
       pEvent->modifiers() != Qt::NoModifier) {
        ImageZoom::mousePressEvent(pEvent);
        return;
    }

    float f = 1.0f / scale();
    float x = pEvent->position().x()*f + rectImage().left();
    float y = pEvent->position().y()*f + rectImage().top();
    float dMin = SELECT_RADIUS;
    select = -1;
    for(size_t i=0; i<pts.size(); i++) {
        float d = std::hypot(pts[i].x-x, pts[i].y-y);
        if(d < dMin) {
            dMin = d;
            select = (int)i;
        }
    }
    if(select>=0) {
        emit new_selection(select);
        update_view();
        QPainter p(&pixmap());
        p.setPen(QPen(pts[select].col));
        float x = (pts[select].x-rectImage().left())*scale();
        float y = (pts[select].y-rectImage().top())*scale();
        p.drawLine(x, y, pEvent->position().x(), pEvent->position().y());
        update();
        ImageZoom::update_status();
    }
    emit new_pos(x,y);
}

void ImagePoints::mouseReleaseEvent(QMouseEvent* pEvent) {
    if(pEvent->button() != Qt::LeftButton ||
       pEvent->modifiers() != Qt::NoModifier) {
        ImageZoom::mouseReleaseEvent(pEvent);
        return;
    }
    select = -1;
    emit new_selection(-1);
    update_view();
    update();
}

void ImagePoints::mouseMoveEvent(QMouseEvent* pEvent) {
    if(pEvent->buttons() != Qt::LeftButton ||
       pEvent->modifiers() != Qt::NoModifier) {
        ImageZoom::mouseMoveEvent(pEvent);
        return;
    }
    float f = 1.0f / scale();
    float x = pEvent->position().x()*f + rectImage().left();
    float y = pEvent->position().y()*f + rectImage().top();
    emit new_pos(x,y);
    if(select>=0) {
        update_view();
        QPainter p(&pixmap());
        p.setPen(pts[select].col);
        x = (pts[select].x-rectImage().left())*scale();
        y = (pts[select].y-rectImage().top())*scale();
        p.drawLine(x, y, pEvent->position().x(), pEvent->position().y());
        update();
    }
    ImageZoom::update_status();
}

void ImagePoints::draw_line(const libNumerics::vector<float>& l) {
    assert(l.nrow()==3);
    const QRect& r = rectImage();
    libNumerics::vector<float> v[4] = {
        libNumerics::vector<float>(r.left(), r.top(), 1),
        libNumerics::vector<float>(r.right(), r.top(), 1),
        libNumerics::vector<float>(r.right(), r.bottom(), 1),
        libNumerics::vector<float>(r.left(), r.bottom(), 1)
    };
    float side[4];
    for(int i=0; i<4; i++)
        side[i] = dot(v[i], l);
    int n=0;
    libNumerics::vector<float> pt[2];
    for(int i=0; i<4 && n<2; i++) {
        if(side[i]==0) {
            pt[n++] = v[i];
            continue;
        }
        if(side[i]*side[(i+1)%4] < 0)
            pt[n++] = cross(l,cross(v[i],v[(i+1)%4]));
    }
    if(n<2)
        return;
    libNumerics::matrix<float> M(3,3);
    float s = (float)scale();
    float f[3*3] = {
        s, 0, -r.left()*s,
        0, s, -r.top()*s,
        0, 0, 1};
    M.read(f);
    for(int i=0; i<2; i++) {
        pt[i] = M*pt[i];
        pt[i] /= pt[i](2);
    }

    update_view();
    QPainter p(&pixmap());
    p.setPen(select>=0? pts[select].col: Qt::green);
    p.drawLine(pt[0](0), pt[0](1), pt[1](0), pt[1](1));
    if(select>=0) {
        libNumerics::vector<float> d(l(0),l(1),0),
            s(pts[select].x,pts[select].y,1);
        pt[0] = cross(l,cross(d,s));
        pt[0]/=pt[0](2);
        float dist = hypot(pt[0](0)-s(0), pt[0](1)-s(1));
        QString str = QString("Dist: %1").arg(dist,0,'f',2);
        message(str);
        pt[0] = M*pt[0];
        pt[0] /= pt[0](2);
        s = M*s;
        s /= s(2);
        p.setPen(Qt::yellow);
        p.drawLine(pt[0](0), pt[0](1), s(0), s(1));
    }
    update();
}

void ImagePoints::update_status(int x, int y) {
    if(select<0) {
        ImageZoom::update_status(x,y);
        return;
    }
    double xd, yd;
    image_from_pixmap_x(x, &xd);
    image_from_pixmap_y(y, &yd);
    float d = std::hypot(xd-pts[select].x, yd-pts[select].y);
    QString str = QString("Dist: %1").arg(d,0,'f',2);
    message(str);
}
