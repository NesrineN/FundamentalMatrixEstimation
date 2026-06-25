#include "MovieControl.h"
#include "MovieViewer.h"

#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QSpinBox>
#include <QtGui/QBitmap>
#include <QtWidgets/QToolTip>
#include <QtCore/QTimer>

/* XPM */
static const char* prev_xpm[] = {
"18 18 3 1",
" 	c None",
".	c #C0C0C0",
"+	c #C93232",
"..................",
".............+.+..",
"............++.+..",
"..........++++.+..",
"........++++++.+..",
"......++++++++.+..",
"....++++++++++.+..",
"..++++++++++++.+..",
".+++++++++++++.+..",
"..++++++++++++.+..",
"....++++++++++.+..",
"......++++++++.+..",
"........++++++.+..",
"..........++++.+..",
"............++.+..",
".............+.+..",
"..................",
".................."};

/* XPM */
static const char* back_xpm[] = {
"18 18 3 1",
" 	c None",
".	c #C0C0C0",
"+	c #C93232",
"..................",
".............+....",
"............++....",
"..........++++....",
"........++++++....",
"......++++++++....",
"....++++++++++....",
"..++++++++++++....",
".+++++++++++++....",
"..++++++++++++....",
"....++++++++++....",
"......++++++++....",
"........++++++....",
"..........++++....",
"............++....",
".............+....",
"..................",
".................."};

/* XPM */
static const char* stop_xpm[] = {
"18 18 3 1",
" 	c None",
".	c #C0C0C0",
"+	c #C93232",
"..................",
"..................",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
".....++...++......",
"..................",
".................."};

/* XPM */
static const char* play_xpm[] = {
"18 18 3 1",
" 	c None",
".	c #C0C0C0",
"+	c #C93232",
"..................",
"....+.............",
"....++............",
"....++++..........",
"....++++++........",
"....++++++++......",
"....++++++++++....",
"....++++++++++++..",
"....+++++++++++++.",
"....++++++++++++..",
"....++++++++++....",
"....++++++++......",
"....++++++........",
"....++++..........",
"....++............",
"....+.............",
"..................",
".................."};

/* XPM */
static const char* next_xpm[] = {
"18 18 3 1",
" 	c None",
".	c #C0C0C0",
"+	c #C93232",
"..................",
".+.+..............",
".+.++.............",
".+.++++...........",
".+.++++++.........",
".+.++++++++.......",
".+.++++++++++.....",
".+.++++++++++++...",
".+.+++++++++++++..",
".+.++++++++++++...",
".+.++++++++++.....",
".+.++++++++.......",
".+.++++++.........",
".+.++++...........",
".+.++.............",
".+.+..............",
"..................",
".................."};

/* XPM */
static const char* loop_xpm[] = {
"18 18 4 1",
" 	c None",
".	c #C0C0C0",
"+	c #C93232",
"@	c #C17474",
"..................",
"..................",
"..................",
"..................",
"..................",
"..+++++++++++@....",
"..+++++++++++++@..",
".............+++..",
"..............++..",
"..............++..",
"....@+.......+++..",
"...++++++++++++@..",
"..+++++++++++@....",
"....@+............",
"..................",
"..................",
"..................",
".................."};

// Constructor
MovieControl::MovieControl(QWidget* parent)
: QWidget(parent), viewer(NULL), timer(NULL)
{
    setAttribute(Qt::WA_QuitOnClose, false);
    QVBoxLayout* layout = new QVBoxLayout;
    setWindowTitle("Control");
    label = new QLabel;
    label->setAlignment(Qt::AlignCenter);
    label->setFont(QFont("courier", 10));
    layout->addWidget(label);

    slider = new QSlider(Qt::Horizontal);
    slider->hide();
    layout->addWidget(slider);

    QHBoxLayout* h = new QHBoxLayout;
    skip = new QSpinBox; skip->setRange(1, 1000); h->addWidget(skip);
    prev = new QToolButton; h->addWidget(prev);
    back = new QToolButton; h->addWidget(back);
    stop = new QToolButton; h->addWidget(stop);
    play = new QToolButton; h->addWidget(play);
    next = new QToolButton; h->addWidget(next);
    loop = new QToolButton; h->addWidget(loop);
    layout->addLayout(h);
    setLayout(layout);

    connect(prev, SIGNAL(clicked()), this, SLOT(slotButton()));
    connect(back, SIGNAL(clicked()), this, SLOT(slotButton()));
    connect(stop, SIGNAL(clicked()), this, SLOT(slotButton()));
    connect(play, SIGNAL(clicked()), this, SLOT(slotButton()));
    connect(next, SIGNAL(clicked()), this, SLOT(slotButton()));
    connect(loop, SIGNAL(clicked()), this, SLOT(slotButton()));

    prev->setToolTip("Previous frame");
    back->setToolTip("Play backwards");
    stop->setToolTip("Stop");
    play->setToolTip("Play");
    next->setToolTip("Next frame");
    loop->setToolTip("Loop");

    prev->setEnabled(false);
    back->setEnabled(false);
    stop->setEnabled(false);
    play->setEnabled(false);
    next->setEnabled(false);
    loop->setEnabled(false);

    back->setCheckable(true);
    play->setCheckable(true);
    prev->setAutoRepeat(true);
    next->setAutoRepeat(true);
    loop->setCheckable(true);

    QPixmap pix1(prev_xpm);
    pix1.setMask(pix1.createHeuristicMask());
    prev->setIcon( QIcon(pix1) );

    QPixmap pix2(back_xpm);
    pix2.setMask(pix2.createHeuristicMask());
    back->setIcon( QIcon(pix2) );

    QPixmap pix3(stop_xpm);
    pix3.setMask(pix3.createHeuristicMask());
    stop->setIcon( QIcon(pix3) );

    QPixmap pix4(play_xpm);
    pix4.setMask(pix4.createHeuristicMask());
    play->setIcon( QIcon(pix4) );

    QPixmap pix5(next_xpm);
    pix5.setMask(pix5.createHeuristicMask());
    next->setIcon( QIcon(pix5) );

    QPixmap pix6(loop_xpm);
    pix6.setMask(pix6.createHeuristicMask());
    loop->setIcon( QIcon(pix6) );

    timer = new QTimer(this);
}

// Destructor
MovieControl::~MovieControl()
{}

// Add movie viewer to control
void MovieControl::add(MovieViewer* v)
{
    connect(v, SIGNAL(changedFrame(MovieViewer*, const QString&)),
            this, SLOT(slotUpdate(MovieViewer*, const QString&)));
    connect(v, SIGNAL(gotFocus(MovieViewer*)),
            this, SLOT(slotFocusIn(MovieViewer*)));
    connect(v, SIGNAL(lostFocus(MovieViewer*)),
            this, SLOT(slotFocusOut(MovieViewer*)));
    connect(v, SIGNAL(hitBound(MovieViewer*)), stop, SIGNAL(clicked()));
    slotFocusIn(v);
}

// Frame changed in controlled viewer, update slider and label
void MovieControl::slotUpdate(MovieViewer* v, const QString& s)
{
    if(viewer == v) {
        slider->setValue(v->at());
        label->setText(s);
        label->repaint();
    }
}

// Controlled viewer got focus
void MovieControl::slotFocusIn(MovieViewer* v)
{
    if(v == NULL || v->frames() <= 1 || v == viewer)
        return;
    timer->stop();
    prev->setEnabled(true);
    back->setEnabled(true);
    stop->setEnabled(false);
    play->setEnabled(true);
    next->setEnabled(true);
    loop->setEnabled(true);
    if(back->isChecked()) back->toggle();
    if(play->isChecked()) play->toggle();
    if(loop->isChecked() != v->looping()) loop->toggle();
    if(viewer) {
        disconnect(skip, SIGNAL(valueChanged(int)), viewer, 0);
        disconnect(viewer, SIGNAL(destroyed()), this, 0);
        disconnect(slider, SIGNAL(sliderMoved(int)), viewer, 0);
    }
    viewer = v;
    slider->setMaximum(v->frames()-1);
    slider->show();
    connect(skip, SIGNAL(valueChanged(int)), v, SLOT(setSkip(int)));
    connect(v, SIGNAL(destroyed()), this, SLOT(slotDestroyed()));
    connect(slider, SIGNAL(sliderMoved(int)), this, SLOT(slotSlider(int)));
    slotUpdate(v, v->windowTitle());
}

// Adapter slot when slider is moved
void MovieControl::slotSlider(int i)
{
    viewer->at((unsigned int)i);
}

// Controlled viewer lost focus
void MovieControl::slotFocusOut(MovieViewer*)
{}

void MovieControl::slotDestroyed()
{
    timer->stop();
    prev->setEnabled(false);
    back->setEnabled(false);
    stop->setEnabled(false);
    play->setEnabled(false);
    next->setEnabled(false);
    viewer = NULL;
    label->clear();
}

void MovieControl::slotButton()
{
    const QObject* which = sender();
    if(which == back || which == play) {
        prev->setEnabled(false);
        back->setEnabled(false);
        stop->setEnabled(true);
        play->setEnabled(false);
        next->setEnabled(false);
        disconnect(timer, SIGNAL(timeout()), 0, 0);
        connect(timer, SIGNAL(timeout()), viewer,
                (which == back) ? SLOT(slotPrev()) : SLOT(slotNext()));
        timer->start(10);
        return;
    }
    if(which == stop) {
        timer->stop();
        prev->setEnabled(true);
        back->setEnabled(true);
        back->setChecked(false);
        stop->setEnabled(false);
        play->setEnabled(true);
        play->setChecked(false);
        next->setEnabled(true);
        return;
    }
    Q_CHECK_PTR(viewer);
    if(which == prev)
        viewer->slotPrev();
    if(which == next)
        viewer->slotNext();
    if(which == loop)
        viewer->setLoop(loop->isChecked());
}
