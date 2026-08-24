#ifndef MOVIECONTROL_H
#define MOVIECONTROL_H

#include <QtWidgets/QWidget>
class QLabel;
class QSlider;
class QToolButton;
class QSpinBox;

class MovieViewer;

class MovieControl : public QWidget
{
    Q_OBJECT
public:
    MovieControl(QWidget* parent=0);
    ~MovieControl();

    void add(MovieViewer*);
private slots:
    void slotUpdate(MovieViewer* v, const QString& s);
    void slotFocusIn(MovieViewer* v);
    void slotFocusOut(MovieViewer*);
    void slotButton();
    void slotDestroyed();
    void slotSlider(int);

private:
    MovieViewer* viewer;
    QLabel* label;
    QSlider* slider;
    QToolButton *prev, *back, *play, *stop, *next, *loop;
    QSpinBox *skip;
    QTimer* timer;
};

#endif
