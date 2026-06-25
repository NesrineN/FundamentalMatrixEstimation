#include "MovieViewer.h"
#include "RawStream.h"

#include <QtWidgets/QLabel>
#include <QtWidgets/QStatusBar>
#include <QtGui/QImage>
#include <QtWidgets/QLayout>

// Static function, read file as movie stream by trying different formats
MovieStream* MovieViewer::read_stream(const QString& name)
{
    MovieStream* stream = new RawStream(name);
    if(stream->frames() <= 0) {
        qDebug(("Failing to read "+name+" as raw file in %1x%2")
               .arg(RawStream::wDefault).arg(RawStream::hDefault)
               .toLocal8Bit().constData());
        delete stream;
        stream = NULL;
    }
    return stream;
}

// Constructor
MovieViewer::MovieViewer(QWidget* parent)
: ImageZoom(parent),
  stream(NULL), ownStream(false), labelTime(NULL), skip(1), loop(false)
{}

// Destructor
MovieViewer::~MovieViewer()
{
    if(ownStream)
        delete stream;
}

// Name of movie file to read. User should call show() after, to display.
void MovieViewer::set(const QString& name)
{
    stream = read_stream(name);
    set(stream);
    ownStream = true;
}

void MovieViewer::set(MovieStream* s)
{
    if(ownStream)
        delete stream;
    ownStream = false;
    stream = s;
    if(stream) {
        if(stream->hasTimeInfo()) {
            // For some reason, the status bar changes the layout margin after
            // addWidget and removeWidget in Qt 4.0
            int margin = statusBar()->layout()->spacing();
            labelTime = new QLabel("(00) 00:00:00.000", statusBar());
            statusBar()->addWidget(labelTime, 0);
            statusBar()->layout()->setSpacing(margin); // Restore original margin
        }
        add_key(Qt::Key_PageUp,    SLOT(slotPrev()));
        add_key(Qt::Key_Backspace, SLOT(slotPrev()));
        add_key(Qt::Key_PageDown,  SLOT(slotNext()));
        add_key(Qt::Key_Space,     SLOT(slotNext()));
    }
}

// Number of frames
unsigned int MovieViewer::frames() const
{
    return (stream == NULL) ? 0 : stream->frames();
}

// Width of a frame
int MovieViewer::w() const
{
    return (stream == NULL) ? 0 : stream->w();
}

// Height of a frame
int MovieViewer::h() const
{
    return (stream == NULL) ? 0 : stream->h();
}

// Frame currently displayed
unsigned int MovieViewer::at() const
{
    return (stream == NULL) ? 0 : stream->at();
}

// Fix position at frame `i'
void MovieViewer::at(unsigned int i)
{
    if(stream != NULL && at() != i) {
        stream->at(i);
        slotCurrent();
    }
}

// Try to display previous frame, if there is one
bool MovieViewer::slotPrev()
{
    bool bound = (stream->at() < (unsigned int)skip);
    if(stream == NULL || (!loop && bound)) {
        emit hitBound(this);
        return false;
    }
    unsigned int pos = stream->at() + (bound? stream->frames(): 0) - skip;
    stream->at(pos);
    slotCurrent();
    return true;
}

// Try to display next frame, if there is one
bool MovieViewer::slotNext()
{
    bool bound = (stream->at()+skip >= stream->frames());
    if(stream == NULL || (!loop && bound)) {
        emit hitBound(this);
        return false;
    }
    unsigned int pos = stream->at() + skip - (bound? stream->frames(): 0);
    stream->at(pos);
    slotCurrent();
    return true;
}

// Display current frame
void MovieViewer::slotCurrent()
{
    if(stream == NULL)
        return;
    QImage image = stream->current();
    if(image.isNull())
        return;

    if(labelTime) {
        unsigned int uTime = stream->time();
        unsigned int sec = uTime/1000; uTime -= sec*1000;
        unsigned int min = sec/60;     sec   -= min*60;
        unsigned int hr  = min/60;     min   -= hr*60;
        unsigned int day = hr/24;      hr    -= day*24;
        QString str = QString("(%1) %2:%3:%4.%5")
            .arg(day,  2,10,QChar(u'0'))
            .arg(hr,   2,10,QChar(u'0'))
            .arg(min,  2,10,QChar(u'0'))
            .arg(sec,  2,10,QChar(u'0'))
            .arg(uTime,3,10,QChar(u'0'));
        labelTime->setText(str);
    }

    unsigned int n = QString().number(stream->frames()-1).length();
    QString s = QString("%1 / %2 ").arg(stream->at(),n).arg(stream->frames());
    QString file = stream->name();
    int d1=file.lastIndexOf('/'), d2=file.lastIndexOf('\\');
    if(d1<d2) d1=d2;
    if(d1 != -1)
        file.remove(0, d1+1);
    s += file;
    s += QString(" (%1x%2)").arg(image.width()).arg(image.height());
    slotLoadImage(image, s);
    emit changedFrame(this, s);
    repaint();
}

// Overload with signal sent
void MovieViewer::focusInEvent(QFocusEvent* e)
{
    ImageZoom::focusInEvent(e);
    emit gotFocus(this);
}

// Overload with signal sent
void MovieViewer::focusOutEvent(QFocusEvent* e)
{
    ImageZoom::focusOutEvent(e);
    emit lostFocus(this);
}
