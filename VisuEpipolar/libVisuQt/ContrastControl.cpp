#include "ContrastControl.h"

ContrastControl::ContrastControl(QWidget* parent)
: QSpinBox(parent),
  _tabColormap(new uchar[256])
{
    setRange(1, 255);
    setFont(QFont("fixed",14));
    setFixedSize(QSize(65, 30));
    connect(this, SIGNAL(valueChanged(int)),
            this, SLOT(slot_quantization_change(int)));
}

ContrastControl::~ContrastControl()
{
  delete [] _tabColormap;
}

/* Slot called when the quantization step changes */
void ContrastControl::slot_quantization_change(int iQuantization)
{
  const int iAdjust = (iQuantization>>1);
  for(int i = iAdjust-1; i >= 0; i--)
    _tabColormap[i] = 0;
  int iMax = 256 - iAdjust;
  for(int i = iAdjust; i < iMax; i++)
    _tabColormap[i] = (uchar)(iQuantization * ((i+iAdjust)/iQuantization));
  for(int i = iMax; i < 256; i++)
    _tabColormap[i] = 255;
  emit colormap_changed(_tabColormap); // Send signal
}
