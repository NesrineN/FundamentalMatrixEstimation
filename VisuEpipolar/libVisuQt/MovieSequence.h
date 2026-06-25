#ifndef MOVIESEQUENCE_H
#define MOVIESEQUENCE_H

#include "MovieStream.h"
#include <vector>

class MovieSequence : public MovieStream
{
public:
    MovieSequence();
    virtual ~MovieSequence();
    
    void append(const QString& str);
    unsigned int at() const { return MovieStream::at(); }
    virtual void at(unsigned int i);
    virtual QImage current();
private:
    std::vector<QString> sequence;
};

#endif
