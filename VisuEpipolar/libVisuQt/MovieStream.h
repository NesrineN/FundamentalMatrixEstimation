#ifndef MOVIESTREAM_H
#define MOVIESTREAM_H

// For support of files >2GB
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#endif
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <fcntl.h>

#include <QtCore/QString>
class QImage;

class MovieStream
{
public:
    MovieStream(const QString& name);
    virtual ~MovieStream() {}

    QString name() const { return movieName; }
    int w() const { return width; }
    int h() const { return height; }
    unsigned int frames() const { return nFrames; }
    unsigned int at() const { return pos; }
    virtual bool hasTimeInfo() const { return false; }
    virtual unsigned int time() const { return 0; }

    virtual void at(unsigned int i)=0;
    virtual QImage current()=0;

    MovieStream& operator++() { at(at()+1); return *this; }
    MovieStream& operator--() { at(at()-1); return *this; }

protected:
    QString movieName;
    int width, height;
    unsigned int nFrames, pos;
};

inline MovieStream::MovieStream(const QString& name)
: movieName(name), width(0), height(0), nFrames(0), pos(0)
{}

#endif
