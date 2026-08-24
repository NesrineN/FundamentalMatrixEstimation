#include "ImagePoints.h"
#include "SynchroF.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <iostream>

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    if(argc>3) {
        std::cerr << "Usage: " << argv[0] << " imageL imageR" << std::endl;
        return 1;
    }

    ImagePoints* IL = new ImagePoints;
    if(argc<2) {
        QString nameFile = QFileDialog::getOpenFileName(0, "Load imageL");
        if(nameFile.isEmpty())
            return 0;
        IL->slotLoadImage(nameFile.toUtf8().data());
    } else
        IL->slotLoadImage(argv[1]);
    IL->show();

    ImagePoints* IR = new ImagePoints;
    if(argc<3) {
        QString nameFile = QFileDialog::getOpenFileName(0, "Load imageR");
        if(nameFile.isEmpty())
            return 0;
        IR->slotLoadImage(nameFile.toUtf8().data());
    } else
        IR->slotLoadImage(argv[2]);
    IR->show();

    SynchroF* s = new SynchroF(IL, IR);
    s->show();
    return a.exec();
}
