#include "ImageZoom.h"

#include <QtGui/QImage>
#include <QtGui/QImageWriter>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>
#include <QtGui/QBitmap>
#include <QtGui/QShortcut>
#include <QtGui/QTransform>

#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtGui/QKeyEvent>

#include <QtWidgets/QStatusBar>
#include <QtWidgets/QLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QApplication>

#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QHeaderView>

#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QColorDialog>

#include <algorithm>
#include <iostream>
#include <cmath>

// Zoom cursor
#define CURSOR_W 16
#define CURSOR_H 16
static unsigned char cursor_mask[] = {
   0xff, 0x07, 0xff, 0x07, 0xff, 0x07, 0x07, 0x00, 0xf7, 0x0f, 0xf7, 0xef,
   0xf7, 0xef, 0x77, 0xee, 0x77, 0xee, 0xf7, 0xef, 0xf7, 0xef, 0xf0, 0xff,
   0x00, 0xf8, 0xe0, 0xff, 0xe0, 0xff, 0xe0, 0xff};
static unsigned char cursor[] = {
   0xff, 0x07, 0x01, 0x04, 0xfd, 0x07, 0x05, 0x00, 0xf5, 0x0f, 0x15, 0xe8,
   0xd5, 0xeb, 0x55, 0xaa, 0x55, 0xaa, 0xd5, 0xab, 0x17, 0xa8, 0xf0, 0xb7,
   0x00, 0x88, 0xe0, 0x8f, 0x20, 0x80, 0xe0, 0xff};

static uchar gray(uchar r, uchar g, uchar b) {
    return (r*11 + g*16 + b*5) / 32;
}

void convert_gray(QImage& im) {
    const int w=im.width(), h=im.height();
    for(int i=0; i<h; i++) {
        uchar* p = im.scanLine(i);
        for(int j=0; j<w; j++) {
            uchar g = gray(p[0],p[1],p[2]);
            *p++ = g;
            *p++ = g;
            *p++ = g;
            p++; // Skip alpha
        }
    }
}

/* Constructor */
ImageZoom::ImageZoom(QWidget* parent, Qt::WindowFlags wFlags)
: QWidget(parent, wFlags),
  _keys(),
  _bHadFocus(true),
  _cFactor(0),
  _pImage(NULL),
  _values(NULL),
  _pPixmap(NULL),
  _pStatusBar(NULL),
  _pStatus(NULL),
  _rectImage(0,0,0,0),
  _maskColor(Qt::black),
  _bMaskImage(false),
  _bGray(false),
  _xClick(-1), _yClick(-1), _xCur(-1), _yCur(-1)
{
  // Build status bar
    _pStatusBar = new QStatusBar(this);
    _pStatus = new QLabel(_pStatusBar);
    _pStatus->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    _pStatus->setFont(QFont("fixed", 8));
    _pStatus->setFixedHeight(fontMetrics().height() + 7);
    _pStatusBar->setSizeGripEnabled(false);
    _pStatusBar->addWidget(_pStatus, 1);
    _pStatusBar->layout()->setContentsMargins(0,0,0,0);
    _pStatusBar->adjustSize();

    add_key(Qt::Key_H,      SLOT(help_keys()));
    add_key(Qt::Key_Left,   SLOT(pan_left()));
    add_key(Qt::Key_Up,     SLOT(pan_up()));
    add_key(Qt::Key_Right,  SLOT(pan_right()));
    add_key(Qt::Key_Down,   SLOT(pan_down()));
    add_key(Qt::Key_Return, SLOT(resize_original()));
    add_key(Qt::Key_Tab,    SLOT(resize_best()));
    add_key(Qt::Key_S,      SLOT(save_image()));
    add_key(Qt::Key_B,      SLOT(choose_mask_color()));
    add_key(Qt::Key_M,      SLOT(toggle_mask()));
    add_key(Qt::Key_G,      SLOT(toggle_gray()));
    add_key(Qt::Key_I,      SLOT(toggle_status()));
    add_key(Qt::Key_Q, QCoreApplication::instance(), SLOT(quit()));

    setMinimumSize(50, 1+_pStatusBar->height()); // Status remains visible
    QPalette palette;
    palette.setColor(backgroundRole(), Qt::cyan);
    setPalette(palette);
    adjustSize(); // Set size to sizeHint()
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_OpaquePaintEvent);
}

/* Destructor */
ImageZoom::~ImageZoom()
{
    delete _pImage;
    free(_values);
    delete _pPixmap;
}

/* Real estate occupied by status bar */
int ImageZoom::heightStatus() const
{ return (!isVisible()||_pStatusBar->isVisible())? _pStatusBar->height(): 0; }

/* Inherited function redefinition */
QSize ImageZoom::sizeHint() const
{
    if(_pImage) {
        QSize sz = (*QApplication::screens().begin())->size();
        int w = std::min(_pImage->width(), 9*sz.width() /10);
        int h = std::min(_pImage->height(),9*sz.height()/10);
        return QSize(w, h+heightStatus());
    }
    return minimumSize();
}

/* Try to load a float image */
bool ImageZoom::load_fimage(const char* , QImage& image)
{
    return false;
    int w, h, ok;
    float* v = 0; //load_rim(const_cast<char*>(sFileName), &h, &w, &ok);
    if(! ok) {
        free(v);
        return false;
    }
    free(_values); _values = v;
    image = QImage(w, h, QImage::Format_Indexed8);
    image.setColorCount(256);
    for(int i = 0; i < 256; i++)
        image.setColor(i, qRgb(i,i,i));
    for(int i = 0; i < h; i++) {
        uchar* pLine = image.scanLine(i);
        for(int j = 0; j < w; j++, v++)
            if(*v < 0) pLine[j] = 0;
            else if(*v > 255.0f) pLine[j] = 255;
            else pLine[j] = (uchar)(*v);
    }
    return true;
}

/* Load the corresponding image (can be NULL), and return success status */
bool ImageZoom::slotLoadImage(const char* sFileName)
{
    QImage image;
    if(sFileName == NULL ||
       ! (image.load(sFileName) || load_fimage(sFileName, image))) {
        unloadImage();
        return (sFileName == NULL);
    }
    // Set window caption
    QString sBaseName(sFileName);
    if(sBaseName.lastIndexOf('/') != -1)
        sBaseName.remove(0, sBaseName.lastIndexOf('/')+1);
    QString sDimension = QString(" (%1x%2)").arg(image.width())
                                            .arg(image.height());
    slotLoadImage(image, sBaseName.append(sDimension));
    return true;
}

void ImageZoom::slotLoadImage(const QImage& image, QString caption)
{
    free(_values); _values = 0;
    setWindowTitle(caption);
    QImage im = image.convertToFormat(QImage::Format_ARGB32);
    // Initialize image and pixmap
    if(_pImage == NULL) {
        _pImage = new QImage(im);
        resize(sizeHint()); // Not adjustSize() that sets max related to screen
    } else
        *_pImage = im;
    loadImage(); // Virtual function, do nothing here
    update_status();
    update_view(_rectImage.x(), _rectImage.y());
    update();
}

void ImageZoom::slotLoadImage(float* v, int w, int h, const QString& caption)
{
    free(_values); _values = v;
    setWindowTitle(caption);
    QImage image(w, h, QImage::Format_Indexed8);
    image.setColorCount(256);
    for(int i = 0; i < 256; i++)
        image.setColor(i, qRgb(i,i,i));
    float fMin = v[0], fMax = v[0];
    for(int i = w*h-1; i > 0; i--)
        if(v[i] < fMin)
            fMin = v[i];
        else if(v[i] > fMax)
            fMax = v[i];
    float a = 1.0f, b = 0;
    if(/*(fMin < 0 || fMax >= 256.0f) &&*/ fMin != fMax) {
        a = 256.0f / (fMax - fMin);
        b = - a * fMin;
    }

    for(int i = 0; i < h; i++) {
        uchar* pLine = image.scanLine(i);
        for(int j = 0; j < w; j++, v++) {
            float val = a * (*v) + b;
            if(val < 0) pLine[j] = 0;
            else if(val > 255.0f) pLine[j] = 255;
            else pLine[j] = (uchar)val;
        }
    }

    // Initialize image and pixmap
    if(_pImage == NULL) {
        _pImage = new QImage(image);
        resize(sizeHint());
    } else
        *_pImage = image;
    loadImage(); // Virtual function, do nothing here
    update_status();
    update_view(_rectImage.x(), _rectImage.y());
    update();
}

void ImageZoom::unloadImage()
{
    delete _pImage; _pImage = NULL;
    free(_values); _values = NULL;
    delete _pPixmap; _pPixmap = NULL;
    repaint();
}

void ImageZoom::slotChangeColormap(const uchar* tabColormap)
{
    if(! _pImage || ! _pPixmap) return;
    Q_ASSERT(_pImage->depth() == 8);
    for(int i = 0; i < 256; i++)
        _pImage->setColor(i, qRgb(tabColormap[i],tabColormap[i],tabColormap[i]));
    if(isVisible()) {
        update_view();
        update();
    }
}

/* Access function */
QImage& ImageZoom::image() const
{
    if(_pImage == 0)
        _pImage = new QImage;
    return *_pImage;
}

///////////////////////////////////////////////////////////////////////
///////////////////////// Event handlers

/* Mouse button press event:
- Ctrl-LeftClick: zoom in
- Ctrl-RightClick: zoom out
- Ctrl-MidClick: prepare to pan
- Shift-LeftClick: measure distance
- Shift-RightClick: measure distance and print it on console */
void ImageZoom::mousePressEvent(QMouseEvent* pEvent)
{
    if(! _pImage) return;
    if(! hadFocus() && pEvent->modifiers() == Qt::ControlModifier) {
        _bHadFocus = true;
        set_zoom_cursor(true);
        return;
    }
    _bHadFocus = true;
    if(pEvent->button() == Qt::MiddleButton &&   // Prepare to pan
       pEvent->modifiers() == Qt::ControlModifier) {
        _xClick = rshift(pEvent->position().x());
        _yClick = rshift(pEvent->position().y());
        return;
    }
    if((pEvent->button()==Qt::LeftButton || pEvent->button()==Qt::RightButton)&&
       pEvent->modifiers() == Qt::ShiftModifier) {  // Distance measurement
        _xClick = pEvent->position().x();
        _yClick = pEvent->position().y();
        _xCur = _yCur = -1;
        return;
    }
    if(pEvent->modifiers() == Qt::ControlModifier) {
        if(pEvent->button() == Qt::LeftButton) {
            zoom_in(pEvent->position().x(), pEvent->position().y());
            return;
        }
        if(pEvent->button() == Qt::RightButton) {
            zoom_out(pEvent->position().x(), pEvent->position().y());
            return;
        }
    }
    emit clicked(pEvent->pos());
}

/* Pan image or info about the pixel being viewed */
void ImageZoom::mouseMoveEvent(QMouseEvent* pEvent)
{
    if(! _pImage) return;
    if(pEvent->modifiers() == Qt::ControlModifier &&
       pEvent->buttons() == Qt::MiddleButton) { // Pan
        int newX = rshift(pEvent->position().x());
        int newY = rshift(pEvent->position().y());
        pan(_xClick-newX, _yClick-newY);
        _xClick = newX;
        _yClick = newY;
        return;
    }
    if(_xClick>=0 && pEvent->modifiers() == Qt::ShiftModifier && // Measure dist
       (pEvent->buttons() & (Qt::LeftButton | Qt::RightButton))) {
        _xCur = pEvent->position().x();
        _yCur = pEvent->position().y();
        repaint();
        return;
    }
    if(pEvent->modifiers() != 0)
        emit clicked(pEvent->pos());
    update_status(pEvent->position().x(), pEvent->position().y());
}

/* Mouse button released */
void ImageZoom::mouseReleaseEvent(QMouseEvent* pEvent)
{
    if(_xCur >= 0) {
        if(pEvent->button() == Qt::RightButton) {
            double d = hypot(_xCur-_xClick,_yCur-_yClick)/scale();
            QString str = QString(" (%1)").arg(d,0,'f',2);
            std::cout << (measure()+str).toStdString() << std::endl;
        }
        _xCur = _yCur = _xClick = _yClick = -1;
    } else
        emit unclicked(pEvent->pos());
}

/* Wheel mouse rotated. Stangely, two identical events are sent each time.
We only process one. */
void ImageZoom::wheelEvent(QWheelEvent* pEvent)
{
    static bool bAccept = false;
    if(! (bAccept = !bAccept))
        return;
    if(_pImage && pEvent->modifiers() == Qt::ControlModifier) {
        if(pEvent->angleDelta().y() > 0)
            zoom_in(pEvent->position().x(), pEvent->position().y());
        else
            zoom_out(pEvent->position().x(), pEvent->position().y());
        pEvent->accept();
    }
}

/* The widget has been resized */
void ImageZoom::resizeEvent(QResizeEvent*)
{
    QPoint o = origin_draw();
    _pStatusBar->setGeometry(o.x(), o.y()+height_view() - _pStatusBar->height(),
                             width_view(), _pStatusBar->height());
    if(_pImage) {
        update_view();
        QString str = QString("%1x%2+%3+%4").arg(_rectImage.width())
                                            .arg(_rectImage.height())
                                            .arg(_rectImage.x())
                                            .arg(_rectImage.y());
        message(str);
    }
}

/* Display information about current zoom factor */
void ImageZoom::display_zoom()
{
    QString s("Scale: ");
    s.append(QString().setNum( (_cFactor > 0)? lshift(1): 1 ));
    s.append(":");
    s.append(QString().setNum( (_cFactor > 0)? 1: rshift(1) ));
    message(s);
}

/* Change mouse cursor to standard (FALSE) or zoom (TRUE) */
void ImageZoom::set_zoom_cursor(bool b)
{
    if(b) {
        static const QBitmap zoom_bitmap =
            QBitmap::fromData(QSize(CURSOR_W, CURSOR_H), ::cursor);
        static const QBitmap zoom_mask =
            QBitmap::fromData(QSize(CURSOR_W, CURSOR_H), ::cursor_mask);
        // Not static, otherwise segmentation fault in QCursor::~QCursor
        QCursor zoom_cursor(zoom_bitmap, zoom_mask);
        setCursor(zoom_cursor);
        display_zoom();
    } else {
        setCursor(Qt::ArrowCursor);
        update_status();
    }
}

/* Change cursor if zoom */
void ImageZoom::keyPressEvent(QKeyEvent* pEvent)
{
    if(_pPixmap &&
       pEvent->key() == Qt::Key_Control &&
       pEvent->modifiers() == Qt::ControlModifier)
        set_zoom_cursor(true);
    else
        pEvent->ignore();
}

/* Various actions */
void ImageZoom::keyReleaseEvent(QKeyEvent* pEvent)
{
    if(_pPixmap && pEvent->key() == Qt::Key_Control) // Set original cursor
        set_zoom_cursor(false);
    else
        pEvent->ignore();
}

void ImageZoom::add_key(int key, const QObject* receiver, const char* member)
{
    QShortcut* s = new QShortcut(QKeySequence(key), this);
    connect(s, SIGNAL(activated()), receiver, member);
    _keys.push_back(std::pair<int, QString>(key, QString(member).mid(1u)));
}

void ImageZoom::add_key(int key, const char* member)
{ add_key(key, this, member); }

void ImageZoom::simulate_key(QTableWidgetItem* item)
{
    qDebug("simulate_key...");
    if(! item) return;
    QString txt = item->text();
    std::vector<std::pair<int,QString> >::const_iterator it;
    for(it = _keys.begin(); it != _keys.end(); ++it)
        if(QKeySequence(it->first).toString() == txt || it->second == txt)
            break;
    if(it != _keys.end()) {
        //        qDebug(QString("key found in table: %1").arg(it->first));
        //QKeyEvent* ev1 = new QKeyEvent(QEvent::ShortcutOverride,
        //                               it->first, 0);
        QShortcutEvent* ev1 = new QShortcutEvent(QKeySequence(it->first), 0);
        QCoreApplication::postEvent(this, ev1);
    }
}

void ImageZoom::help_keys()
{
    QTableWidget* table = new QTableWidget(this);
    table->setWindowFlags(Qt::Window);
    table->setAttribute(Qt::WA_DeleteOnClose);
    table->setColumnCount(2);
    table->setHorizontalHeaderLabels(QStringList() << "Key" << "Action");
    table->verticalHeader()->hide();
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    std::vector<std::pair<int, QString> >::const_iterator it;
    for(it = _keys.begin(); it != _keys.end(); ++it) {
        int i = table->rowCount();
        table->insertRow(i);
        table->setItem(i, 0,
                       new QTableWidgetItem(QKeySequence(it->first).toString()));
        table->setItem(i, 1, new QTableWidgetItem(it->second));
    }
    connect(table, SIGNAL(itemDoubleClicked(QTableWidgetItem*)),
            this, SLOT(simulate_key(QTableWidgetItem*)));
    table->setWindowTitle("Keys");
    table->show();
}

void ImageZoom::pan_left()
{    pan(std::min(-1,-bshift(width_view(), _cFactor+1)), 0); }
void ImageZoom::pan_up()
{    pan(0, std::min(-1,-bshift(height_view()-heightStatus(), _cFactor+1))); }
void ImageZoom::pan_right()
{    pan(std::max(1,bshift(width_view(), _cFactor+1)), 0); }
void ImageZoom::pan_down()
{    pan(0, std::max(1,bshift(height_view()-heightStatus(), _cFactor+1))); }
void ImageZoom::resize_original()
{
    _cFactor = 0; // Restore original image
    _rectImage.setX(0); _rectImage.setY(0);
    display_zoom();
    if(size() != sizeHint())
        resize(sizeHint());
    else {
        update_view();
        update();
    }
}
void ImageZoom::resize_best()
{
    QSize oldSize = size();
    int w = _pImage? lshift(_pImage->width()): 0;
    int h = (_pImage? lshift(_pImage->height()):0) + heightStatus();
    QSize sz = (*QApplication::screens().begin())->size();
    w = std::min(w, 9*sz.width()/10);
    h = std::min(h, 9*sz.height()/10);
    resize(w, h);
    QResizeEvent* e = new QResizeEvent(QSize(w,h), oldSize);
    QCoreApplication::postEvent(this, e);
    
}

void ImageZoom::save_image()
{
    if(! _pPixmap)
        return;
    QList<QByteArray> formats = QImageWriter::supportedImageFormats();
    QString all("All images("), filter;
    for(QList<QByteArray>::const_iterator it=formats.begin();
        it!=formats.end(); ++it) {
        filter += it->toUpper().constData();
        filter += "(*.";
        filter += it->constData();
        filter += ");;";
        all += "*.";
        all += it->constData();
        all += " ";
    }
    all.replace(all.size()-1,1,");;");
    filter.remove(filter.size()-2, 2);
    filter = all + filter;
    static QString name = QDir::currentPath() + "/" + "snapshot.png";
    QString nameFile = QFileDialog::getSaveFileName(this, "Save image", name,
                                                    filter);
    if(! nameFile.isEmpty())
        if(_pPixmap->save(nameFile))
            name = nameFile;
        else
            QMessageBox::warning( this, "Save failed", "Error saving file" );
}

void ImageZoom::choose_mask_color()
{
    QColor col = QColorDialog::getColor(_maskColor, this);
    if(col.isValid()) {
        _maskColor = col;
        _bMaskImage = true;
        update_view();
        update();
    }
}

void ImageZoom::toggle_mask()
{
    if(!_pImage) return;
    _bMaskImage = !_bMaskImage;
    update_view();
    update();
}

void ImageZoom::toggle_gray()
{
    if(!_pImage) return;
    _bGray = ! _bGray;
    update_view();
    update();
    update_status();
}

void ImageZoom::toggle_status()
{
    if(_pStatusBar->isVisible())
        _pStatusBar->hide();
    else
        _pStatusBar->show();
    update_view();
    update();
}

QString ImageZoom::measure() const {
    double x0, y0, x1, y1;
    image_from_pixmap_x(_xClick, &x0);
    image_from_pixmap_y(_yClick, &y0);
    image_from_pixmap_x(_xCur,   &x1);
    image_from_pixmap_y(_yCur,   &y1);
    QString signx((x1>x0)? "+": "");
    QString signy((y1>y0)? "+": "");
    return QString("(%1,%2)->(%3,%4): %5%6 %7%8")
                   .arg(x0).arg(y0).arg(x1).arg(y1)
                   .arg(signx).arg(x1-x0)
                   .arg(signy).arg(y1-y0);
}

/* Draw the image */
void ImageZoom::paintEvent(QPaintEvent* pEvent)
{
    if(! _pPixmap)
        return;
    QPainter painter(this);
    painter.setClipRect(pEvent->rect());
    painter.setBrush(palette().brush(QPalette::Window).color());
    painter.drawRect(origin_draw().x(),origin_draw().y(),width(),height());
    painter.drawPixmap(origin_draw(), *_pPixmap);
    if(_xCur >= 0) {
        painter.setPen(Qt::white);
        painter.drawLine(_xClick, _yClick, _xCur, _yCur);
        double d = hypot(_xCur-_xClick,_yCur-_yClick)/scale();
        painter.drawText(_xClick, _yClick, QString("%1").arg(d,0,'f',2));
        if(_pStatusBar->isVisible()) {
            _pStatus->setText(measure());
            _pStatusBar->clearMessage();
            _pStatusBar->update();
        }
    }
}

///////////////////////////////////////////////////////////////////////
///////////////////////// Custom functions

/* Zoom in around pixel of coordinates (x,y) in the current view */
void ImageZoom::zoom_in(int x, int y)
{
    // Prevent large zooms showing less than 1 pixel in the whole widget
    const int w = bshift(width_view(), _cFactor+1);
    const int h = bshift(height_view()-heightStatus(), _cFactor+1);
    if(w == 0 || h == 0)
        return;
    ++ _cFactor;
    x = _rectImage.left() + rshift(x-origin_draw().x());
    y = _rectImage.top()  + rshift(y-origin_draw().y());
    update_view(x, y);
    update();
    display_zoom();
}

/* Zoom out around pixel of coordinates (x,y) in the current view */
void ImageZoom::zoom_out(int x, int y)
{
    // Prevent small zoom showing the whole image in half the widget size
    const int w = bshift(width_view(), _cFactor-1);
    const int h = bshift(height_view()-heightStatus(), _cFactor-1);
    if(w > 2*_pImage->width() && h > 2*_pImage->height())
        return;
    x = _rectImage.left() - rshift(x-origin_draw().x());
    y = _rectImage.top()  - rshift(y-origin_draw().y());
    -- _cFactor;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    update_view(x, y);
    update();
    display_zoom();
}

/* Pan the image view. Warning: `dx' and `dy' in original image coordinates */ 
void ImageZoom::pan(int dx, int dy)
{
    if(_pImage == 0 || (dx == 0 && dy == 0))
        return;
    int x = _rectImage.left() + dx;
    int y = _rectImage.top() + dy;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    const int w = _rectImage.width();
    const int h = _rectImage.height();
    if(x+w > _pImage->width()) x = _pImage->width()-w;
    if(y+h > _pImage->height()) y = _pImage->height()-h;
    if(x != _rectImage.x() || y != _rectImage.y()) {
        update_view(x, y);
        repaint();
        update_status();
        QString str = QString("%1x%2+%3+%4").arg(_rectImage.width())
                                            .arg(_rectImage.height())
                                            .arg(_rectImage.x())
                                            .arg(_rectImage.y());
        message(str);
    }
}

/* Transform x in original image coordinate */
int ImageZoom::image_from_pixmap_x(int x, double* exact) const
{
    if(x < 0) x = 0;
    else if(x >= width_view()) x = width_view()-1;
    if(exact)
        *exact = x / scale() + _rectImage.left();
    x = rshift(x) + _rectImage.left();
    if(x >= _pImage->width()) {
        x = _pImage->width()-1;
        if(exact) *exact = _pImage->width();
    }
    return x;
}

/* Transform y in original image coordinate */
int ImageZoom::image_from_pixmap_y(int y, double* exact) const
{
    if(y < 0) y = 0;
    if(y+heightStatus() >= height_view())
        y = height_view() - heightStatus() - 1;
    if(exact)
        *exact = y / scale() + _rectImage.top();
    y = rshift(y) + _rectImage.top();
    if(y >= _pImage->height()) {
        y = _pImage->height()-1;
        if(exact) *exact = _pImage->height();
    }
    return y;
}

/* Return value at pixel (x,y) */
QString ImageZoom::value_at(double x, double y) const
{
    if(x >= _pImage->width()) x = _pImage->width()-1.0;
    if(y >= _pImage->height()) y = _pImage->height()-1.0;
    Q_ASSERT(_pImage->valid((int)x, (int)y));
    QString str;
    if(_values)
        str=QString("%1").arg(_values[_pImage->width()*(int)y + (int)x]);
    else if(_bGray)
        str=QString("%1").arg(gray((int)x, (int)y),3);
    else {
        QColor c = _pImage->pixel(x,y);
        str=QString("%1 %2 %3").arg(c.red(),3).arg(c.green(),3).arg(c.blue(),3);
    }
    return str;
}

/* Return intensity value at pixel (x,y) */
int ImageZoom::gray(int x, int y) const
{
    return gray(x, y, _pImage);
}

/* Return intensity value at pixel (x,y) */
int ImageZoom::gray(int x, int y, const QImage* im)
{
    QRgb rgb=im->pixel(x,y);
    return ::gray(qRed(rgb), qGreen(rgb), qBlue(rgb));
}

/* Update status bar to reflect info about pixel (x,y).
Warning: coordinates in current view */
void ImageZoom::update_status(int x, int y)
{
    if(! _pImage) return;
    double dx, dy;
    x = image_from_pixmap_x(x, &dx);
    y = image_from_pixmap_y(y, &dy);
    QString str = QString("(%1,%2) ").arg(x,3).arg(y,3);
    str += value_at(dx, dy);
    _pStatus->setText(str);
    _pStatusBar->clearMessage();
    if(_pStatusBar->isVisible())
        _pStatusBar->repaint();
}

/* Update status bar with info about current pixel */
void ImageZoom::update_status()
{
    QPoint pos = mapFromGlobal( QCursor::pos() );
    update_status(pos.x(), pos.y());
}

/* Display temporary message in the status bar */
void ImageZoom::message(const QString& str)
{
    if(_pStatusBar->isVisible())
        _pStatusBar->showMessage(str, 5000);
}

// Perform own zoom out of image
void ImageZoom::pixmap_zoom_out()
{
    const int w = _pPixmap->width(), h = _pPixmap->height();
    QImage image(w, h, _bGray? QImage::Format_Grayscale8: QImage::Format_RGB32);
    const int Bpp = _pImage->depth() >> 3; // Bytes per pixel
    const int s = 1 << (-_cFactor), stride = s*Bpp;
    for(int i = 0; i < h; i++) {
        const uchar* in = _pImage->scanLine(s*i+_rectImage.y()) +
            Bpp*_rectImage.x();
        uchar* out = image.scanLine(i);
        for(int j = 0; j < w; j++, in += stride)
            if(_bGray) {
                uchar g = ::gray(in[0],in[1],in[2]);
                *out++ = g;
            } else {
                for(int k = 0; k < Bpp; k++)
                    *out++ = in[k];
            }
    }
    *_pPixmap = QPixmap::fromImage(image);
}

/* Build the pixmap corresponding to _rectImage in the image */
void ImageZoom::pixmap_from_image()
{
    if(_bMaskImage)
        _pPixmap->fill(_maskColor);
    else if(_cFactor < 0)
        pixmap_zoom_out();
    else {
        QImage image = _pImage->copy(_rectImage);
        if(_bGray)
            convert_gray(image);
        if(_cFactor == 0)
            *_pPixmap = QPixmap::fromImage(image);
        else {
            QPixmap pixmap( QPixmap::fromImage(image) );
            QTransform m;
            m.scale(scale(), scale());
            *_pPixmap = pixmap.transformed(m);
        }
    }
}

/* Compute the number of pixels in the original image necessary
to deduce `iDimension' pixels in the pixmap */
int ImageZoom::dim_image_from_pixmap(int iDimension) const
{
    int iInc = (_cFactor > 0)? lshift(1)-1: 0;
    return (iDimension > 0)? rshift(iDimension + iInc): 0;
}

/* Compute the number of pixels that could be deduced from `iDimension'
pixels in the original image */
int ImageZoom::dim_pixmap_from_image(int iDimension) const
{
    return (iDimension > 0)? lshift(iDimension): 0;
}

/* Update the part of the image being viewed. The top left corner
is (x,y) (image coordinates). */
void ImageZoom::update_view(int x, int y)
{
    if(! _pImage) return;
    // Ensure at least one pixel is visible
    if(x >= _pImage->width())
        x = _pImage->width()-1;
    if(y >= _pImage->height())
        y = _pImage->height()-1;
    int w = dim_image_from_pixmap(width_view());
    int h = dim_image_from_pixmap( height_view()-heightStatus() );
    if(x+w > _pImage->width())  w = _pImage->width() - x;
    if(y+h > _pImage->height()) h = _pImage->height() - y;
    _rectImage.setRect(x, y, w, h); // Rect being viewed (image coordinates)
    w = dim_pixmap_from_image(w); h = dim_pixmap_from_image(h);
    if(_pPixmap)
        *_pPixmap = QPixmap(w, h);
    else
        _pPixmap = new QPixmap(w, h);
    pixmap_from_image(); // Build pixmap
}
