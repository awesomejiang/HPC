#ifndef VEC3_H
#define VEC3_H

#include <cmath>

template<typename T>
class Vec3{
public:
	Vec3(): x{0}, y{0}, z{0} {}
	Vec3(T const &x, T const &y, T const &z)
	: x{x}, y{y}, z{z} {}

	Vec3<T> operator*(T const &d){
		return Vec3<T>{x*d, y*d, z*d};
	}

	Vec3<T> &norm(){
		return *this = *this*(1/std::sqrt(x*x + y*y + z*z));
	}

	T x,y,z;
};

template<typename T>
inline Vec3<T> operator+(Vec3<T> const &a, Vec3<T> const &b){
	return Vec3<T>{a.x+b.x, a.y+b.y, a.z+b.z};
}

template<typename T>
inline Vec3<T> operator-(Vec3<T> const &a, Vec3<T> const &b){
	return Vec3<T>{a.x-b.x, a.y-b.y, a.z-b.z};
}

template<typename T>
inline T operator*(Vec3<T> const &a, Vec3<T> const &b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<typename T>
inline Vec3<T> operator*(T const &d, Vec3<T> const &b){
	return Vec3<T>{b.x*d, b.y*d, b.z*d};
}

#endif