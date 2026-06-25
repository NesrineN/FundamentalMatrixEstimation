#ifndef IMAGEZOOM_H
#define IMAGEZOOM_H

#include <QtWidgets/QWidget>
class QImage;
class QPixmap;
class QStatusBar;
class QLabel;
class QTableWidgetItem;

#include <vector>

class ImageZoom : public QWidget {
    Q_OBJECT
public:
    ImageZoom(QWidget* parent=0, Qt::WindowFlags wFlags=Qt::WindowFlags());
    virtual ~ImageZoom();
    virtual QSize sizeHint() const;
    const QImage* original_image() const { return _pImage; }

    const QPixmap& display() const { return *_pPixmap; }
public slots:
    virtual bool slotLoadImage(const char* sFileName);
    virtual void slotLoadImage(const QImage& image, QString caption=QString());
    virtual void slotLoadImage(float* v, int w, int h, const QString& caption);
    void slotChangeColormap(const uchar* tabColormap);

signals:
    void clicked(const QPoint& pos);
    void unclicked(const QPoint& pos);
protected: // Events handlers
    virtual void wheelEvent(QWheelEvent*);
    virtual void paintEvent(QPaintEvent*);
    virtual void resizeEvent(QResizeEvent*);
    virtual void mousePressEvent(QMouseEvent*);
    virtual void mouseMoveEvent(QMouseEvent*);
    virtual void mouseReleaseEvent(QMouseEvent*);
    virtual void loadImage() {}
    virtual void unloadImage();
protected: // Key events
    virtual void keyPressEvent(QKeyEvent*);
    virtual void keyReleaseEvent(QKeyEvent*);
    std::vector<std::pair<int, QString> > _keys;
    void add_key(int key, const QObject* receiver, const char* member);
    void add_key(int key, const char* member);
protected slots:
    void help_keys();
    void simulate_key(QTableWidgetItem*);
    void pan_left();
    void pan_up();
    void pan_right();
    void pan_down();
    void resize_original();
    void resize_best();
    void save_image();
    void choose_mask_color();
    void toggle_mask();
    void toggle_gray();
    void toggle_status();

private: // Dirty tricks with focus
    bool _bHadFocus;
protected:
    void enterEvent(QEvent*) { _bHadFocus = hasFocus(); }
    void focusOutEvent(QFocusEvent*) { set_zoom_cursor(false); }
    bool hadFocus() const // Had keyboard focus last time mouse entered?
    { return _bHadFocus; }

protected: // Zoom and update
    void set_zoom_cursor(bool b);
    void zoom_in(int xCenter, int yCenter); // Zoom in around point
    void zoom_out(int xCenter, int yCenter); // Zoom out
    void pan(int dx, int dy); // Warning: original image coordinates
    virtual void update_status(int x, int y);
    virtual void update_status();
    virtual void message(const QString& str);
    QStatusBar* statusBar() { return _pStatusBar; }
    int heightStatus() const;
    virtual void pixmap_from_image(); // Build pixmap corresponding to _rectImage
    virtual void pixmap_zoom_out();
    void update_view(int x, int y);  // View window of top left corner (x,y)
    void update_view() { update_view(_rectImage.x(), _rectImage.y()); }
    static int bshift(int i, char shift); // bit shift
    int lshift(int i) const { return bshift(i, -_cFactor); }
    int rshift(int i) const { return bshift(i, +_cFactor); }

public: //protected: // Coordinates conversions
    virtual QPoint origin_draw() const { return QPoint(0,0); }
    virtual int width_view() const { return width(); }
    virtual int height_view() const { return height(); }
    virtual int dim_image_from_pixmap(int iDimension) const;
    virtual int dim_pixmap_from_image(int iDimension) const;
    int image_from_pixmap_x(int x, double* exact = 0) const;
    int image_from_pixmap_y(int y, double* exact = 0) const;
    virtual QString value_at(double x, double y) const;
    int gray(int x, int y) const;
    static int gray(int x, int y, const QImage* im);

protected: // Access functions
    QImage& image() const;
    const float* values() const { return _values; }
    QPixmap& pixmap() { return *_pPixmap; }
    const QRect& rectImage() const { return _rectImage; }
    char factor() const { return _cFactor; }
    double scale() const;
    bool masked() const { return _bMaskImage; }
protected:
    bool load_fimage(const char* sFileName, QImage& img);
    void display_zoom();
    QString measure() const;
    char _cFactor; // 2^_cFactor is the zoom factor
    mutable QImage* _pImage; // Original image
    float* _values; // Original pixel values if float image
    QPixmap* _pPixmap; // Current pixmap (visible in the widget)
    QStatusBar* _pStatusBar;
    QLabel* _pStatus; // Status label
    QRect _rectImage; // Rectangle of the original image displayed
    QColor _maskColor; // Color of mask
    bool _bMaskImage; // Display image as black rectangle or not
    bool _bGray; // Display image in gray scale
    int _xClick, _yClick; // Position of cursor during pan (image coordinates)
    int _xCur, _yCur; // Current cursor position
};

// Safe right bit shift
inline int ImageZoom::bshift(int i, char shift)
{
    if(shift >= 0)
        return i >> shift;
    return i  << (-shift);
}

inline double ImageZoom::scale() const
{
    if(_cFactor >= 0)
        return static_cast<double>(lshift(1));
    return 1.0 / bshift(1, _cFactor);
}

void convert_gray(QImage& im);

#endif
