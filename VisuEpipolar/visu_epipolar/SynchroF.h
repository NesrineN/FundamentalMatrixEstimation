#ifndef SYNCHROF_H
#define SYNCHROF_H

#include "ImagePoints.h"
class QLineEdit;
class QLabel;

class SynchroF : public QWidget {
    Q_OBJECT
public:
    SynchroF(ImagePoints* I1, ImagePoints* I2, QWidget* parent=0);
protected slots:
    void select(int i); // The point selection has changed
    void pos(float x, float y); // The pointer moved, draw epipolar line
    void read_F_coeff(); // One coefficient of F was edited
    void transpose(); // Transpose matrix
    void loadMatch();
    void loadF();
private:
    ImagePoints* I[2];
    libNumerics::matrix<float> F;
    QLineEdit* coeffFEdit[3*3];
    QLabel* labelError;
    void fill_label_error();
};

#endif
