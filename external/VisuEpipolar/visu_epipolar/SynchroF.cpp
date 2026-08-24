#include "SynchroF.h"
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtGui/QDoubleValidator>
#include <fstream>

SynchroF::SynchroF(ImagePoints* I1, ImagePoints* I2, QWidget* parent)
: QWidget(parent), F(3,3) {
    float v[3*3] = {
        0, 0, 0,
        0, 0, -1,
        0, 1, 0
    };
    F.read(v);
    I[0] = I1; I[1] = I2;
    connect(I1, &ImagePoints::new_selection, this, &SynchroF::select);
    connect(I2, &ImagePoints::new_selection, this, &SynchroF::select);
    connect(I1, &ImagePoints::new_pos, this, &SynchroF::pos);
    connect(I2, &ImagePoints::new_pos, this, &SynchroF::pos);

    QGridLayout* layout = new QGridLayout(this);
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++) {
            coeffFEdit[3*i+j] = new QLineEdit;
            coeffFEdit[3*i+j]->setValidator(new QDoubleValidator);
            coeffFEdit[3*i+j]->setText(QString("%1").arg(F(i,j)));
            connect(coeffFEdit[3*i+j], &QLineEdit::editingFinished,
                    this, &SynchroF::read_F_coeff);
            layout->addWidget(coeffFEdit[3*i+j],i,j);
        }

    QCheckBox* check = new QCheckBox("transpose");
    layout->addWidget(check,3,0);
    connect(check, &QCheckBox::checkStateChanged, this, &SynchroF::transpose);
    labelError = new QLabel;
    labelError->setAlignment(Qt::AlignHCenter);

    labelError->setToolTip("Average epipolar errors");
    layout->addWidget(labelError,3,1);
    fill_label_error();

    QVBoxLayout* layoutButtons = new QVBoxLayout;
    QPushButton* pushLoadMatch = new QPushButton("Load points");
    QPushButton* pushLoadF = new QPushButton("Load F");
    layoutButtons->addWidget(pushLoadMatch);
    layoutButtons->addWidget(pushLoadF);
    layout->addLayout(layoutButtons,3,2);
    connect(pushLoadMatch, &QPushButton::clicked, this, &SynchroF::loadMatch);
    connect(pushLoadF,     &QPushButton::clicked, this, &SynchroF::loadF);
}

void SynchroF::select(int i) {
    QObject* s = sender();
    int j = I[0]==s? 1: 0;
    I[j]->set_selection(i);
}

void SynchroF::pos(float x, float y) {
    QObject* s = sender();
    int i = I[0]==s? 1: 0;
    libNumerics::vector<float> v(x,y,1);
    libNumerics::matrix<float> F0 = i==0? F: F.t();
    v = F0*v;
    I[i]->draw_line(v);
}

void SynchroF::read_F_coeff() {
    QObject* s = sender();
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            if(s == coeffFEdit[3*i+j]) {
                double d = QLocale().toDouble(coeffFEdit[3*i+j]->text());
                F(i,j) = (float)d;
                fill_label_error();
                return;
            }
}

void SynchroF::transpose() {
    F = F.t();
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            coeffFEdit[3*i+j]->setText(QString("%1").arg(F(i,j)));
    fill_label_error();
}

void SynchroF::fill_label_error() {
    int n = (int)I[0]->pts.size();
    assert(n==(int)I[1]->pts.size());
    float dl=0, dr=0;
    libNumerics::matrix<float> Ft = F.t();
    for(int i=0; i<n; i++) {
        libNumerics::vector<float>
            xl(I[0]->pts[i].x, I[0]->pts[i].y, 1.0f),
            xr(I[1]->pts[i].x, I[1]->pts[i].y, 1.0f);
        libNumerics::vector<float> linel=F*xr, liner=Ft*xl;
        float d = std::abs(dot(xl,linel));
        dl += d/hypot(linel(0),linel(1));
        dr += d/hypot(liner(0),liner(1));
    }
    dl /= (float)n;
    dr /= (float)n;
    labelError->setText(QString("Errors:\n%1(left)\n%2(right)")
                        .arg(dl).arg(dr));
}

void SynchroF::loadMatch() {
    QString nameFile = QFileDialog::getOpenFileName(this, "Load points");
    if(nameFile.isEmpty())
        return;
    std::ifstream f(nameFile.toUtf8().data());
    if(! f.is_open()) {
        QMessageBox::warning(this, "Open failed", "Failed opening file");
        return;
    }
    std::vector<Point> pts1,pts2;
    int ignored=0;
    while(! f.eof()) {
        std::string str;
        std::getline(f, str);
        std::istringstream is(str);
        QColor col(rand()%255,rand()%255,rand()%255);
        Point p1(0,0,col), p2(0,0,col);
        is >> p1.x >> p1.y >> p2.x >> p2.y;
        if(is.fail()) {
            if(!f.eof() || !str.empty())
                ++ignored;
            continue;
        }
        pts1.push_back(p1);
        pts2.push_back(p2);
    }
    QMessageBox::information(this, "Reading points",
                             QString("Points: %1\nIgnored: %2")
                             .arg(pts1.size()).arg(ignored));
    if(pts1.empty())
        return;
    std::swap(I[0]->pts,pts1);
    std::swap(I[1]->pts,pts2);
    I[0]->set_selection(-1);
    I[1]->set_selection(-1);
    fill_label_error();
}

void SynchroF::loadF() {
    QString nameFile = QFileDialog::getOpenFileName(this, "Load matrix F");
    if(nameFile.isEmpty())
        return;
    std::ifstream f(nameFile.toUtf8().data());
    if(! f.is_open()) {
        QMessageBox::warning(this, "Open failed", "Failed opening file");
        return;
    }
    libNumerics::matrix<float> F0(3,3);
    f >> F0;
    if(f.fail()) {
        QMessageBox::warning(this, "Reading matrix", "Failed reading matrix");
        return;
    }
    F = F0;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            coeffFEdit[3*i+j]->setText(QString("%1").arg(F(i,j)));
    fill_label_error();
}
