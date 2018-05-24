#ifndef VEC3_H
#define VEC3_H

#include <cuda.h>

template<typename T>
class Vec3{
public:
	__device__ __host__ Vec3(): x{0}, y{0}, z{0} {}
	__device__ __host__ Vec3(T const &x, T const &y, T const &z)
	: x{x}, y{y}, z{z} {}

	__device__ __host__ Vec3<T> operator*(T const &d){
		return Vec3<T>{x*d, y*d, z*d};
	}
	
	__device__ __host__ Vec3<T> &norm(){
		return *this = *this * rsqrt(x*x + y*y + z*z);
	}

	T x,y,z;
};

template<typename T>
__device__ __host__ Vec3<T> operator+(Vec3<T> const &a, Vec3<T> const &b){
	return Vec3<T>{a.x+b.x, a.y+b.y, a.z+b.z};
}

template<typename T>
__device__ __host__ Vec3<T> operator-(Vec3<T> const &a, Vec3<T> const &b){
	return Vec3<T>{a.x-b.x, a.y-b.y, a.z-b.z};
}

template<typename T>
__device__ __host__ T operator*(Vec3<T> const &a, Vec3<T> const &b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<typename T>
__device__ __host__ Vec3<T> operator*(T const &d, Vec3<T> const &b){
	return Vec3<T>{b.x*d, b.y*d, b.z*d};
}

#endif