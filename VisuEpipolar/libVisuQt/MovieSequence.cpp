#include "MovieSequence.h"
#include <QtGui/QImage>
#include <iostream>

MovieSequence::MovieSequence()
: MovieStream("")
{}

MovieSequence::~MovieSequence()
{}

void MovieSequence::append(const QString& str)
{
    if(sequence.empty())
        movieName = str;
    sequence.push_back(str);
    nFrames = sequence.size();
    if(width == 0 || height == 0) {
        QImage im(str);
        width = im.width();
        height = im.height();
    }
}

void MovieSequence::at(unsigned int i)
{
    pos = i;
    movieName = sequence[i];
}

QImage MovieSequence::current()
{
    QImage im(sequence[pos]);
    if(im.isNull())
        std::cerr << "Unable to load image " << sequence[pos].toStdString()
                  << std::endl;
    return im;
}
