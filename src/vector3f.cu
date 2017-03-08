#include "vector3f.h"

std::ostream& operator<< (std::ostream &o, const Math::vector3f &a)
{
  return a.toString(o);
}

namespace Math
{

}