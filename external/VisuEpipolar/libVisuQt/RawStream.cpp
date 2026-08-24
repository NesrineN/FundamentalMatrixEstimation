#include "RawStream.h"
#include <QtGui/QImage>

int RawStream::wDefault = 640;
int RawStream::hDefault = 480;

// Constructor
RawStream::RawStream(const QString& name)
: MovieStream(name), file(NULL)
{
    if(! init(wDefault, hDefault) && file) {
        (void)fclose(file);
        file = NULL;
    }
}

// Constructor
RawStream::RawStream(const QString& name, int w, int h)
: MovieStream(name), file(NULL)
{
    if(! init(w, h) && file) {
        (void)fclose(file);
        file = NULL;
    }
}

// Try to initialize with given width and height
bool RawStream::init(int w, int h)
{
    file = fopen(name().toLocal8Bit().constData(), "r");
    if(file == NULL) return false;
    width = w; height = h;
    if(fseeko(file, 0, SEEK_END)) return false;
    off_t size = ftello(file);
    rewind(file);
    off_t n = size / (3*width*height);
    if(n*(3*width*height) != size) return false;
    nFrames = (int)n;
    return true;
}

// Put at given frame
void RawStream::at(unsigned int i)
{
    if(file == NULL) return;
    if(i >= nFrames-1)
        pos = nFrames-1;
    else
        pos = i;
    off_t n = pos*off_t(3*width*height);
    (void)fseeko(file, n, SEEK_SET);
}

// Return current frame
QImage RawStream::current()
{
    if(! file)
        return QImage();
    size_t s = 3*width*height;
    unsigned char* buf = new unsigned char[s];
    QImage image;
    off_t where = ftello(file);
    if(fread(buf, s, 1, file) == 1) {
        image = QImage(width, height, QImage::Format_RGB32);
        unsigned char *pr = buf;
        unsigned char *pg = pr+width*height;
        unsigned char *pb = pg+width*height;    
        for(int i = 0; i < height; i++) {
            uint* pLine = (uint*)image.scanLine(i);
            for(int j = 0; j < width; j++) {
                unsigned char r = *pr++;
                unsigned char g = *pg++;
                unsigned char b = *pb++;
                pLine[j] = qRgb(r, g, b);
            }
        }
    }
    fseeko(file, where, SEEK_SET);
    delete [] buf;
    return image;
}
