#ifndef RAWSTREAM_H
#define RAWSTREAM_H

#include "MovieStream.h"
class QString;

class RawStream : public MovieStream
{
public:
    static int wDefault; // = 640;
    static int hDefault; // = 480;
public:
    RawStream(const QString& name);
    RawStream(const QString& name, int w, int h);

    unsigned int at() const { return MovieStream::at(); }
    virtual void at(unsigned int i);
    virtual QImage current();

private:
    FILE* file;
    bool init(int w, int h);
};

#endif
