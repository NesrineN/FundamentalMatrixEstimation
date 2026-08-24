#ifndef MOVIEVIEWER_H
#define MOVIEVIEWER_H

#include "ImageZoom.h"
class MovieStream;
class QLabel;

class MovieViewer : public ImageZoom
{
    Q_OBJECT
public:
    static MovieStream* read_stream(const QString& name);

    MovieViewer(QWidget* parent=0);
    virtual ~MovieViewer();

    virtual void show() { ImageZoom::show(); slotCurrent(); }

    virtual void set(const QString& name);
    virtual void set(MovieStream* s);
    unsigned int frames() const;
    int w() const;
    int h() const;
    unsigned int at() const;
    bool looping() const { return loop; }
signals:
    void changedFrame(MovieViewer* self, const QString& nameFrame);
    void gotFocus(MovieViewer* self);
    void lostFocus(MovieViewer* self);
    void hitBound(MovieViewer* self);
public slots:
    void at(unsigned int i);
    virtual bool slotPrev();
    virtual void slotCurrent();
    virtual bool slotNext();
    void setSkip(int s) { Q_ASSERT(s>=1); skip = s%frames(); }
    void setLoop(bool b) { loop = b; }
protected:
    virtual void focusInEvent(QFocusEvent* e);
    virtual void focusOutEvent(QFocusEvent* e);
    MovieStream* stream;
    bool ownStream;
    QLabel* labelTime;
    int skip;
    bool loop;
};

#endif
