#ifndef CONTRASTCONTROL_H
#define CONTRASTCONTROL_H

#include <QtWidgets/QSpinBox>

class ContrastControl : public QSpinBox {
  Q_OBJECT;
public:
  ContrastControl(QWidget* pParent =0);
  virtual ~ContrastControl();
signals:
  void colormap_changed(const uchar*);
private slots:
  void slot_quantization_change(int iQuantization);
private:
  uchar* _tabColormap;
};

#endif
